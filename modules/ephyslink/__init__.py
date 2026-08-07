"""
ephyslink — one session object, two languages, one file.

The problem it solves: an analysis pipeline that produces a new file at every stage, in a
format only one language can read, with axis order that has to be guessed at the other end.

    from ephyslink import load_kilosort, Session

    session = load_kilosort("/data/2026-04-14_11-46-55")
    session.add_table("units", unit_table)
    session.add_array("psth", matrix, dims=["unit", "bin"])
    session.log("responder test", alpha=0.05)
    session.save("analysis/2026-04-14.h5")

    session = Session.load("analysis/2026-04-14.h5")     # Python
                                                          # or, in Julia:
    #   using EphysLink
    #   s = load_session("analysis/2026-04-14.h5")
    #   s.arrays["psth"]        # same axis order, same names

Two typed views over one file format:

* `SessionOE` — continuous traces: LFP, MEP, EEG. `(channel, sample)`.
* `SessionKS` — spike-sorted output: spike times, clusters, templates, cluster table.

`arrays`, `tables` and `events` are open namespaces. Anything a pipeline stage produces can be
attached to the session and travels with it, which is the point — the alternative is a
directory of loose `.npy` files whose relationship to each other is in your head.

New to this? `modules/QUICKSTART.md` is the tour.

The file format is specified in `../FORMAT.md`, and the Julia implementation is
`modules/EphysLink.jl/`. If the two ever disagree, `FORMAT.md` is right and both are wrong.
Run `selftest.py` and `../EphysLink.jl/test/roundtrip.jl` after touching either.
"""

from .format import FORMAT_NAME, FORMAT_VERSION, describe_file, read_session_dict, write_session
from .kilosort import load_kilosort, read_params
from .openephys import (
    describe_recording, find_recording, load_openephys, memmap_continuous, read_structure,
)
from .results import ResultSet, read_results
from .session import Session, SessionKS, SessionOE, SessionResults

__all__ = [
    "Session", "SessionOE", "SessionKS", "SessionResults",
    "read_results", "ResultSet",
    "load_kilosort", "read_params",
    "load_openephys", "memmap_continuous", "describe_recording", "find_recording",
    "read_structure",
    "write_session", "read_session_dict", "describe_file",
    "FORMAT_NAME", "FORMAT_VERSION",
]
__version__ = "1.0.0"
