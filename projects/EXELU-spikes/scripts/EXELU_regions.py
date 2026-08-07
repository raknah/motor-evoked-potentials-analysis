"""
Turning a probe channel into a depth, and a depth into a brain region.

The probe is a single shank descending through visual cortex into hippocampus. Kilosort
gives every spike an estimated depth along that shank. The only anatomy available is a
handful of landmark *channels* the experimenter identified from the LFP and marked in
`session_particulars.txt`:

    thetaCh   the channel at which the theta rhythm appears — taken as the cortex /
              hippocampus boundary
    CA1Ch     CA1 pyramidal layer
    V1L4Ch    layer 4 of V1
    DGsCh     dentate gyrus, suprapyramidal blade
    DGiCh     dentate gyrus, infrapyramidal blade

Channel numbering is **1-based** (MATLAB), confirmed by the professor, so channel *n* sits
at depth `(n - 1) * pitch`. Getting this wrong shifts every boundary by one contact
(20 um) and has previously moved units across the cortex/hippocampus line.
"""

from __future__ import annotations

import numpy as np

from EXELU_config import CHANNEL_PITCH_UM

LANDMARK_SUFFIX = "Ch"

CORTEX = "CTX"
HIPPOCAMPUS = "HPC"


def channel_to_depth_um(channel: float, pitch_um: float = CHANNEL_PITCH_UM) -> float:
    """Depth below the top contact, in micrometres, of a **1-based** channel number."""
    return (float(channel) - 1.0) * pitch_um


def landmarks_um(
    particulars: dict[str, str | None], pitch_um: float = CHANNEL_PITCH_UM
) -> dict[str, float]:
    """Every populated `*Ch` entry in the particulars, converted to depth."""
    return {
        key: channel_to_depth_um(value, pitch_um)
        for key, value in particulars.items()
        if key.endswith(LANDMARK_SUFFIX) and value not in (None, "")
    }


def boundary_um(particulars: dict[str, str | None], pitch_um: float = CHANNEL_PITCH_UM) -> float:
    """Depth of the cortex/hippocampus boundary — the theta channel."""
    marks = landmarks_um(particulars, pitch_um)
    if "thetaCh" not in marks:
        raise KeyError(
            "session_particulars.txt has no thetaCh, so no region assignment is possible. "
            "Every region-level claim is blocked for this session."
        )
    return marks["thetaCh"]


def region_of_depth(depth_um, boundary: float) -> np.ndarray:
    """Shallower than the theta channel is cortex; deeper is hippocampus.

    A hard threshold with no buffer zone. Units sitting within one contact of the boundary
    are genuinely ambiguous and are flagged separately by `near_boundary`.
    """
    return np.where(np.asarray(depth_um) < boundary, CORTEX, HIPPOCAMPUS)


def near_boundary(depth_um, boundary: float, tolerance_um: float = CHANNEL_PITCH_UM) -> np.ndarray:
    """True for units whose region assignment would flip under a one-contact error."""
    return np.abs(np.asarray(depth_um) - boundary) <= tolerance_um


# =============================================================================
# the same three questions, asked of an ephyslink SessionKS
#
# The landmark channels live in `session.meta["particulars"]`, attached when the session was
# assembled. These wrappers exist so no analysis module has to know that.
# =============================================================================

def particulars(session) -> dict:
    return session.meta.get("particulars", {}) or {}


def landmarks(session, pitch_um: float = CHANNEL_PITCH_UM) -> dict[str, float]:
    """Every populated `*Ch` entry for this session, converted to depth."""
    return landmarks_um(particulars(session), pitch_um)


def boundary(session, pitch_um: float = CHANNEL_PITCH_UM) -> float:
    """Depth of the cortex/hippocampus boundary for this session."""
    return boundary_um(particulars(session), pitch_um)


def check_pitch(session, expected_um: float = CHANNEL_PITCH_UM) -> float:
    """Verify the assumed contact spacing against the probe geometry Kilosort recorded.

    Every depth-to-region boundary is `(channel - 1) * pitch`, so a wrong pitch moves every
    boundary. Checked once, at load, rather than assumed.
    """
    positions = session.array("channel_positions")
    observed = float(np.median(np.diff(np.unique(positions[:, 1]))))
    if not np.isclose(observed, expected_um):
        raise ValueError(
            f"CHANNEL_PITCH_UM is {expected_um} µm but channel_positions.npy implies "
            f"{observed} µm. Every depth-to-region boundary depends on this."
        )
    return observed
