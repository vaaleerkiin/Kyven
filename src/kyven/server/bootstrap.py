"""Windows-safe entry point that detaches Kyven from a host application's DLL search path."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def reset_inherited_dll_directory() -> None:
    """Clear SetDllDirectory state inherited from Nuke before importing PyTorch."""
    if os.name != "nt":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_dll_directory = kernel32.SetDllDirectoryW
    set_dll_directory.argtypes = [ctypes.c_wchar_p]
    set_dll_directory.restype = ctypes.c_int
    if not set_dll_directory(None):
        raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    reset_inherited_dll_directory()
    # The Nuke adapter can be updated with git/ZIP while the private environment
    # still contains an older wheel. Running this checkout's entry point must
    # import this checkout's backend, even under Python's isolated (-I) mode.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("[Kyven] Loading backend", file=sys.stderr, flush=True)
    from kyven.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
