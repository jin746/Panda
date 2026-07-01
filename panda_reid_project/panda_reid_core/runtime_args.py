"""Runtime argument helpers used by inference entry points."""

from types import SimpleNamespace


def build_infer_args_for_inference(
    cfg_path: str,
    out_root: str,
    mode: str,
    sim_th: float,
    verbose: bool = False,
):
    """Build the minimal argument namespace expected by PandaReIDInference."""
    return SimpleNamespace(
        cfg=cfg_path,
        opts=[],
        local_rank=0,
        mode=mode,
        similarity_threshold=sim_th,
        base_threshold=0.2,
        adaptive_threshold_min=0.15,
        adaptive_threshold_max=0.6,
        confidence_threshold=0.2,
        quality_threshold=0.05,
        use_simple_logic=True,
        roi_format="mask",
        mask_root=None,
        batch_size=32,
        verbose=verbose,
        output_root=out_root,
    )

