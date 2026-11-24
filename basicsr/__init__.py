"""Minimal BasicSR runtime bundled with R2UHD.

Only the framework pieces R2UHD needs are present: registry/option handling,
the paired-image data pipeline, L1/FFT/SSIM losses, the model adapter and the
training entry point. Run scripts from this directory; no installation is
required because ``basicsr`` resolves to the local package.
"""

__version__ = '1.0.0'
