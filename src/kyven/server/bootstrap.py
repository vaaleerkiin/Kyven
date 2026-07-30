"""Windows-safe entry point that detaches Kyven from a host application's DLL search path."""

from __future__ import annotations

import os


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
    from kyven.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
