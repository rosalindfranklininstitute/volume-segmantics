"""Per-head student-vs-teacher disagreement.

Heads absent from either side are skipped silently. Weights are
renormalised over the present subset so a 2-head config gets
meaningful weights without the caller doing the renormalisation.

Sign convention: higher value = more disagreement.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch


#: Default per-head weights (uniform). Override via the ``weights``
#: argument to :func:`compute_per_head_disagreement`.
DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS: Dict[str, float] = {
    "semantic": 1.0,
    "boundary": 1.0,
    "distance": 1.0,
    "sdm":      1.0,
}


def compute_per_head_disagreement(
    student_out: Mapping[str, torch.Tensor],
    teacher_out: Mapping[str, torch.Tensor],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """Per-head student-vs-teacher disagreement map.

    Returns a ``(B, 1, H, W)`` (2D) or ``(B, 1, D, H, W)`` (3D) float
    tensor in ``[0, 1]``. Heads absent from either side are skipped.

    Parameters
    ----------
    student_out, teacher_out
        Dicts keyed by head name (``semantic`` / ``boundary`` /
        ``distance`` / ``sdm``). Tensors carry the head's native
        activation: semantic = logits, boundary = logits, distance =
        identity, sdm = tanh-bounded.
    weights
        Optional ``{head_name: float}`` dict. Defaults to
        :data:`DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS`. Unknown names
        are ignored; absent names default to ``1.0``.

    Returns
    -------
    torch.Tensor
        Per-voxel disagreement map. Empty intersections return a
        zeros tensor with shape derived from the first available
        student head.

    Raises
    ------
    ValueError
        ``student_out`` or ``teacher_out`` is empty (no heads). The
        function defensively requires at least one populated head per
        side so the call site can't silently get zero signal.
    """
    if not student_out:
        raise ValueError(
            "compute_per_head_disagreement: student_out has no heads — "
            "nothing to compare against."
        )
    if not teacher_out:
        raise ValueError(
            "compute_per_head_disagreement: teacher_out has no heads — "
            "nothing to compare against."
        )

    eff_weights = (
        dict(weights) if weights else dict(DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS)
    )

    common = sorted(set(student_out.keys()) & set(teacher_out.keys()))
    if not common:
        # Empty intersection — return a zero map with shape derived
        # from any student head so the caller still gets a tensor.
        ref = next(iter(student_out.values()))
        out_shape = (ref.shape[0], 1) + tuple(ref.shape[2:])
        return torch.zeros(out_shape, device=ref.device, dtype=ref.dtype)

    per_head_maps: Dict[str, torch.Tensor] = {}
    for name in common:
        s = student_out[name]
        t = teacher_out[name].detach()
        if name == "semantic":
            # argmax-match per voxel.
            s_arg = s.argmax(dim=1, keepdim=True)
            t_arg = t.argmax(dim=1, keepdim=True)
            d = (s_arg != t_arg).to(s.dtype)
        elif name == "boundary":
            s_p, t_p = torch.sigmoid(s), torch.sigmoid(t)
            d = (s_p - t_p).abs().mean(dim=1, keepdim=True).clamp_(0.0, 1.0)
        elif name == "distance":
            d = (s - t).abs().mean(dim=1, keepdim=True).clamp_(0.0, 1.0)
        elif name == "sdm":
            d = (s - t).abs().mean(dim=1, keepdim=True).clamp_(0.0, 1.0)
        else:
            # Unknown head — skip with no contribution.
            continue
        per_head_maps[name] = d

    if not per_head_maps:
        ref = next(iter(student_out.values()))
        out_shape = (ref.shape[0], 1) + tuple(ref.shape[2:])
        return torch.zeros(out_shape, device=ref.device, dtype=ref.dtype)

    # Weighted sum, renormalised over the present subset.
    active_weights = {
        n: float(eff_weights.get(n, 1.0)) for n in per_head_maps
    }
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0:
        # All-zero weights -> uniform mean fallback so the caller
        # doesn't silently get zeros when they clearly wanted signal.
        stacked = torch.stack(list(per_head_maps.values()), dim=0)
        return stacked.mean(dim=0).clamp_(0.0, 1.0)

    combined: Optional[torch.Tensor] = None
    for n, m in per_head_maps.items():
        contribution = (active_weights[n] / weight_sum) * m
        combined = contribution if combined is None else combined + contribution
    assert combined is not None
    return combined.clamp_(0.0, 1.0)


__all__ = [
    "DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS",
    "compute_per_head_disagreement",
]
