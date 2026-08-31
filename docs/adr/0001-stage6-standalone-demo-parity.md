# Keep the Stage 6 standalone demo semantically aligned with offline replay

The Windows CUDA standalone webcam demo will load the validated Stage 6 bundle and reuse its feature, temporal, calibration, resolver, and history contracts rather than maintain camera-specific decision formulas. Camera capture and tracking remain environment-specific; invalid bundles soft-disable classification with a visible warning and never trigger silent fallback to an older artifact, because such fallback would make Gate 4 provenance unverifiable.
