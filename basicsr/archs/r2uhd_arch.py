"""R2UHD: three-scale bidirectional restoration network for UHD images.

Three independent dual-branch stages; independent H/4 bridges and steps;
spatially encoded K16 centers; complete residual identity with low/high
corrections; stronger H/4 restoration and H/2 decoding; no H/16. Widths
16/24/32, FP32 UHD whole-frame target.

The parameter layout is fixed: ``ARCH_REVISION`` identifies the exact
topology so that a checkpoint can never be loaded into a different network.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

# Stored in every checkpoint as ``_architecture_revision``. It is the guard
# that makes weight mixing between R2UHD and any other architecture loud
# instead of silent. Do not change it unless the parameter layout changes.
ARCH_REVISION = 171


class ChannelNorm(nn.Module):
    """Per-pixel channel normalization; FP32 statistics under autocast."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.float()
            var, mean = torch.var_mean(z, dim=1, keepdim=True, unbiased=False)
            z = (z - mean) * torch.rsqrt(var + 1e-6)
            return (z * self.weight.float() + self.bias.float()).to(x.dtype)


class GatedFFN(nn.Module):
    """Depthwise gated feed-forward network."""
    def __init__(self, dim, expansion):
        super().__init__()
        hidden = int(dim * expansion)
        self.expand = nn.Conv2d(dim, 2 * hidden, 1)
        self.dw = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.project = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        a, b = self.dw(self.expand(x)).chunk(2, dim=1)
        return self.project(F.gelu(a) * b)


class MixedScaleDepthwise(nn.Conv2d):
    """One 3x3 kernel per channel: first half dense, second half dilation=2.

    Same weight/bias shapes and MACs as an ordinary depthwise 3x3, so the
    high-frequency path sees two receptive fields at no extra parameter cost.
    """
    def __init__(self, dim, bias=True):
        super().__init__(dim, dim, 3, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        split = self.in_channels // 2
        if not split:
            return super().forward(x)
        a, b = x.split([split, self.in_channels - split], dim=1)
        ba = self.bias[:split] if self.bias is not None else None
        bb = self.bias[split:] if self.bias is not None else None
        a = F.conv2d(a, self.weight[:split], ba, padding=1, groups=split)
        b = F.conv2d(b, self.weight[split:], bb, padding=2, dilation=2,
                     groups=self.in_channels - split)
        return torch.cat((a, b), dim=1)


class LocalBlock(nn.Module):
    """Spatial residual block used in encoding, guidance, detail restoration and decoding."""
    def __init__(self, dim, expansion, scale):
        super().__init__()
        self.norm1 = ChannelNorm(dim)
        self.mix = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                 nn.Conv2d(dim, dim, 1))
        self.norm2 = ChannelNorm(dim)
        self.ffn = GatedFFN(dim, expansion)
        self.scale1 = nn.Parameter(torch.full((1, dim, 1, 1), scale))
        self.scale2 = nn.Parameter(torch.full((1, dim, 1, 1), scale))

    def forward(self, x):
        x = x + self.scale1.to(x.dtype) * self.mix(self.norm1(x))
        return x + self.scale2.to(x.dtype) * self.ffn(self.norm2(x))


class ComplementaryClusterAttention(nn.Module):
    """Spatially encoded centers plus a complete, complementary residual path.

    B = assignment-broadcast centers; R = V-B; L = pool/upsample(R); H = R-L.
    Output projects V + broadcast(center_delta) + gated low/high corrections.
    L/H describe latent spatial scales, not clean/degraded image semantics.
    """
    record_diagnostics = False

    def __init__(self, dim, heads, prototypes, scale):
        super().__init__()
        if dim % heads or dim % 4:
            raise ValueError('core width must be divisible by heads and four residual groups')
        self.heads = heads
        self.groups = 4
        self.value = nn.Conv2d(dim, dim, 1)
        self.spatial_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.spatial_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-3))
        self.prototypes = nn.Parameter(torch.empty(prototypes, dim))
        nn.init.normal_(self.prototypes, std=1.0 / math.sqrt(dim))
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.center_norm = nn.LayerNorm(dim)
        self.center_qkv = nn.Linear(dim, 3 * dim)
        self.center_project = nn.Linear(dim, dim)
        self.center_scale = nn.Parameter(torch.full((1, 1, dim), scale))
        hidden = max(1, dim // 2)
        self.center_ffn_norm = nn.LayerNorm(dim)
        self.center_ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.normal_(self.center_ffn[-1].weight, std=1e-3)
        nn.init.zeros_(self.center_ffn[-1].bias)
        self.center_ffn_scale = nn.Parameter(torch.full((1, 1, dim), scale))
        self.low_dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.low_project = nn.Conv2d(dim, dim, 1)
        self.hf_dw = MixedScaleDepthwise(dim)
        self.hf_project = nn.Conv2d(dim, dim, 1)
        self.hf_dw2 = MixedScaleDepthwise(dim)
        self.hf_project2 = nn.Conv2d(dim, dim, 1)
        self.band_gate = nn.Sequential(nn.Conv2d(3 * dim, 8, 1), nn.GELU(),
                                       nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.GELU(),
                                       nn.Conv2d(8, 2 * self.groups, 1))
        for projection in (self.low_project, self.hf_project2):
            nn.init.normal_(projection.weight, std=1e-3)
            nn.init.zeros_(projection.bias)
        self.project = nn.Conv2d(dim, dim, 1)
        self.last_diagnostics = None

    @staticmethod
    def split_residual(residual):
        h, w = residual.shape[-2:]
        if min(h, w) > 1:
            low = F.interpolate(F.avg_pool2d(residual, 2), size=(h, w), mode='nearest')
        else:
            low = residual
        return low, residual - low

    def forward(self, x):
        b, c, h, w = x.shape
        value = self.value(x)
        value = value + self.spatial_scale.to(value.dtype) * self.spatial_dw(value)
        flat = value.flatten(2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            unit_v = F.normalize(flat.float(), dim=1, eps=1e-6)
            unit_p = F.normalize(self.prototypes.float(), dim=1, eps=1e-6)
            temperature = self.log_temperature.float().clamp(-4, 4).exp()
            assign = (torch.matmul(unit_p, unit_v) / temperature).softmax(dim=1)
            mass = assign.sum(dim=-1, keepdim=True) + 1e-4
            centers = torch.matmul(assign, flat.float().transpose(1, 2)) / mass
            rebuild = torch.matmul(assign.transpose(1, 2), centers).transpose(1, 2)
        centers = centers.to(x.dtype)
        rebuild = rebuild.to(x.dtype).reshape(b, c, h, w)
        n = centers.shape[1]
        head_dim = c // self.heads
        q, k, v = self.center_qkv(self.center_norm(centers)).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.heads, head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, head_dim).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = (q.float() @ k.float().transpose(-2, -1) / math.sqrt(head_dim)).softmax(dim=-1)
            mixed = (weights @ v.float()).transpose(1, 2).reshape(b, n, c)
        refined = centers + self.center_scale.to(x.dtype) * self.center_project(mixed.to(x.dtype))
        refined = refined + self.center_ffn_scale.to(x.dtype) * self.center_ffn(self.center_ffn_norm(refined))
        with torch.autocast(device_type=x.device.type, enabled=False):
            global_delta = torch.matmul(assign.transpose(1, 2), (refined - centers).float()).transpose(1, 2)
        global_delta = global_delta.to(x.dtype).reshape(b, c, h, w)
        residual = value - rebuild
        low, high = self.split_residual(residual)
        gate_input = torch.cat((low.abs(), high.abs(), rebuild), dim=1)
        gate = torch.sigmoid(self.band_gate(gate_input))
        low_gate, high_gate = gate.chunk(2, dim=1)
        low_gate = low_gate.repeat_interleave(c // self.groups, dim=1)
        high_gate = high_gate.repeat_interleave(c // self.groups, dim=1)
        low_delta = low_gate * self.low_project(F.gelu(self.low_dw(low)))
        high_delta = high_gate * self.hf_project2(F.gelu(self.hf_dw2(
            self.hf_project(F.gelu(self.hf_dw(high))))))
        if self.record_diagnostics:
            with torch.no_grad():
                denom = value.float().square().mean().sqrt().clamp_min(1e-8)
                self.last_diagnostics = {
                    'assignment_entropy': float(-(assign.clamp_min(1e-12).log()*assign).sum(1).mean()
                                                / math.log(assign.shape[1])),
                    'occupancy': (assign.mean((0, 2))).tolist(),
                    'partition_max_error': float((value-(rebuild+low+high)).abs().max()),
                    'low_gate_mean': float(low_gate.mean()),
                    'high_gate_mean': float(high_gate.mean()),
                    'global_delta_rms_ratio': float(global_delta.float().square().mean().sqrt()/denom),
                    'low_delta_rms_ratio': float(low_delta.float().square().mean().sqrt()/denom),
                    'high_delta_rms_ratio': float(high_delta.float().square().mean().sqrt()/denom)}
        # Exact V=B+L+H base; correction branches need not reconstruct raw R.
        return self.project(value + global_delta + low_delta + high_delta)


class GlobalLayer(nn.Module):
    """One independently learned H/8 content and residual update."""
    def __init__(self, dim, heads, prototypes, expansion, scale):
        super().__init__()
        self.read_norm = ChannelNorm(dim)
        self.read_gate = nn.Conv2d(2 * dim, dim, 1)
        self.read_project = nn.Conv2d(dim, dim, 1)
        self.read_scale = nn.Parameter(torch.full((1, dim, 1, 1), scale))
        self.norm1 = ChannelNorm(dim)
        self.attn = ComplementaryClusterAttention(dim, heads, prototypes, scale)
        self.norm2 = ChannelNorm(dim)
        self.ffn = GatedFFN(dim, expansion)
        self.scale1 = nn.Parameter(torch.full((1, dim, 1, 1), scale))
        self.scale2 = nn.Parameter(torch.full((1, dim, 1, 1), scale))

    def forward(self, x, condition):
        gate = torch.sigmoid(self.read_gate(torch.cat((self.read_norm(x), condition), dim=1)))
        x = x + self.read_scale.to(x.dtype) * self.read_project(gate * condition)
        x = x + self.scale1.to(x.dtype) * self.attn(self.norm1(x))
        return x + self.scale2.to(x.dtype) * self.ffn(self.norm2(x))


class BidirectionalBridge(nn.Module):
    """H/4->H/8 high-resolution guidance; H/8->H/4 low-resolution feedback."""
    def __init__(self, detail_dim, core_dim, expansion, scale, detail_depth):
        super().__init__()
        self.high_resolution_guidance_mean = nn.Conv2d(detail_dim, core_dim, 1, bias=False)
        self.high_resolution_guidance_learned = nn.Sequential(
            LocalBlock(detail_dim, expansion, scale), nn.PixelUnshuffle(2),
            nn.Conv2d(4 * detail_dim, core_dim, 1))
        self.guidance_scale = nn.Parameter(torch.full((1, core_dim, 1, 1), scale))
        self.context_fuse = nn.Conv2d(2 * core_dim, core_dim, 1)
        self.condition_block = LocalBlock(core_dim, expansion, scale)
        self.low_resolution_feedback = nn.Sequential(
            nn.Conv2d(core_dim, 4 * detail_dim, 1), nn.PixelShuffle(2))
        self.source_projections = nn.ModuleList([nn.Conv2d(detail_dim, detail_dim, 1) for _ in range(3)])
        self.source_norms = nn.ModuleList([ChannelNorm(detail_dim) for _ in range(3)])
        self.source_logits = nn.Conv2d(3 * detail_dim, 3 * detail_dim, 1)
        self.restore = nn.Sequential(*[LocalBlock(detail_dim, expansion, scale) for _ in range(detail_depth)])
        self.correction = nn.Conv2d(detail_dim, detail_dim, 1)
        nn.init.normal_(self.correction.weight, std=1e-3)
        nn.init.zeros_(self.correction.bias)

    def condition(self, detail, context):
        guidance = self.high_resolution_guidance_mean(F.avg_pool2d(detail, 2))
        guidance = guidance + self.guidance_scale * self.high_resolution_guidance_learned(detail)
        return self.condition_block(self.context_fuse(torch.cat((context, guidance), 1)))

    def fuse_sources(self, detail, feedback, evidence):
        sources = [p(x) for p, x in zip(self.source_projections, (detail, feedback, evidence))]
        logits = self.source_logits(torch.cat([n(x) for n,x in zip(self.source_norms,sources)],1))
        b, _, h, w = logits.shape
        weights = logits.reshape(b, 3, detail.shape[1], h, w).softmax(dim=1)
        fused = sum(weights[:, i] * source for i, source in enumerate(sources))
        return fused

    def forward(self, global_state, detail, evidence, step):
        feedback = self.low_resolution_feedback(global_state)
        fused = self.fuse_sources(detail, feedback, evidence)
        delta = self.correction(self.restore(fused))
        return detail + step.to(delta.dtype) * delta, feedback


class PackDown(nn.Sequential):
    def __init__(self, source, target):
        super().__init__(nn.PixelUnshuffle(2), nn.Conv2d(4 * source, target, 1))


@ARCH_REGISTRY.register()
class R2UHDNet(nn.Module):
    """R2UHD restoration network: three feed-forward stages, no cross-stage sharing."""
    def __init__(self, *, widths=(16,24,32), num_stages=3, global_depth=2, num_heads=4,
                 prototypes=16, ffn_expansion=2.0, residual_scale=0.1, encoder2_depth=2, detail_depth=3, decoder4_depth=2, decoder2_depth=3):
        super().__init__()
        if len(widths) != 3 or any(type(v) is not int or v < 2 for v in widths):
            raise ValueError('widths must contain three integers >= 2')
        for name, value in dict(num_stages=num_stages, global_depth=global_depth,
                                num_heads=num_heads, prototypes=prototypes, encoder2_depth=encoder2_depth,
                                detail_depth=detail_depth, decoder4_depth=decoder4_depth, decoder2_depth=decoder2_depth).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if prototypes < 2:
            raise ValueError('at least two prototypes are required')
        if widths[2] % num_heads or widths[2] % 4:
            raise ValueError('core width must be divisible by heads and four residual groups')
        if not math.isfinite(ffn_expansion) or ffn_expansion < 1:
            raise ValueError('ffn_expansion must be finite and >= 1')
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('residual_scale must be finite and positive')
        fine, detail, core = widths
        self.num_stages, self.global_depth, self.prototypes = num_stages, global_depth, prototypes
        self.scale_factor = 8
        self.register_buffer('_architecture_revision', torch.tensor(ARCH_REVISION, dtype=torch.int64))
        self.stem = nn.Conv2d(12, fine, 1)
        self.encoder2 = nn.Sequential(*[LocalBlock(fine, ffn_expansion, residual_scale) for _ in range(encoder2_depth)])
        self.down4 = PackDown(fine, detail)
        self.encoder4 = LocalBlock(detail, ffn_expansion, residual_scale)
        self.down8 = PackDown(detail, core)
        self.context = nn.Sequential(nn.Conv2d(fine+detail+core, core, 1), nn.GELU(),
                                     nn.Conv2d(core, core, 1))
        self.stages = nn.ModuleList([
            nn.ModuleList([GlobalLayer(core, num_heads, prototypes, ffn_expansion, residual_scale)
                           for _ in range(global_depth)]) for _ in range(num_stages)])
        self.bridges = nn.ModuleList([
            BidirectionalBridge(detail, core, ffn_expansion, residual_scale, detail_depth) for _ in range(num_stages)])
        self.detail_steps = nn.Parameter(torch.full((num_stages, 1, detail, 1, 1), residual_scale))
        self.fuse4 = nn.Conv2d(3*detail, detail, 1)
        self.decoder4 = nn.Sequential(*[LocalBlock(detail, ffn_expansion, residual_scale) for _ in range(decoder4_depth)])
        self.up2 = nn.Sequential(nn.Conv2d(detail, 4*fine, 1), nn.PixelShuffle(2))
        self.fuse2 = nn.Conv2d(2*fine, fine, 1)
        self.detail_project = nn.Conv2d(fine, fine, 1)
        self.detail_gate = nn.Sequential(nn.Conv2d(fine, fine, 3, padding=1, groups=fine),
                                        nn.GELU(), nn.Conv2d(fine, 1, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(*[LocalBlock(fine, ffn_expansion, residual_scale) for _ in range(decoder2_depth)])
        self.head = nn.Conv2d(fine, 12, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        revision = state_dict.get(prefix+'_architecture_revision')
        if revision is None or revision.numel()!=1 or revision.item()!=ARCH_REVISION:
            raise RuntimeError(
                f'R2UHD requires an architecture revision {ARCH_REVISION} checkpoint; '
                'weights from other revisions are incompatible')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def set_diagnostics(self, enabled=True):
        for stage in self.stages:
            for layer in stage:
                layer.attn.record_diagnostics = bool(enabled)
                layer.attn.last_diagnostics = None

    def diagnostics(self):
        return [[layer.attn.last_diagnostics for layer in stage] for stage in self.stages]

    def forward(self, x):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<1:
            raise ValueError('Expected nonempty BCHW RGB input')
        h, w = x.shape[-2:]
        ph, pw = (-h)%8, (-w)%8
        if ph or pw:
            x = F.pad(x, (0,pw,0,ph), mode='reflect' if ph<h and pw<w else 'replicate')
        e2 = self.encoder2(self.stem(F.pixel_unshuffle(x, 2)))
        e4 = self.encoder4(self.down4(e2))
        g = self.down8(e4)
        context = self.context(torch.cat((F.avg_pool2d(e2,4), F.avg_pool2d(e4,2), g),dim=1))
        d = e4
        # Each stage owns its H/8 layers and complete H/4 bridge. The loop only
        # traverses distinct feed-forward modules; no updater is called twice.
        for index, (stage, bridge) in enumerate(zip(self.stages, self.bridges)):
            condition = bridge.condition(d, context)
            for layer in stage:
                g = layer(g, condition)
            d, feedback = bridge(g, d, e4, self.detail_steps[index])
            del condition
        d = self.decoder4(self.fuse4(torch.cat((feedback,d,e4),dim=1)))
        base = self.fuse2(torch.cat((self.up2(d),e2),dim=1))
        detail = self.detail_project(e2)
        fused = base + self.detail_gate(base-detail)*detail
        residual = F.pixel_shuffle(self.head(self.decoder2(fused)),2)
        return (x+residual)[...,:h,:w]

    @torch.no_grad()
    def test(self, input):
        return self(input)
