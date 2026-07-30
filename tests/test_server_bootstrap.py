from __future__ import annotations

import os
import unittest

from kyven.server.bootstrap import reset_inherited_dll_directory


class ServerBootstrapTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows DLL search behavior")
    def test_reset_inherited_dll_directory_is_idempotent(self) -> None:
        reset_inherited_dll_directory()
        reset_inherited_dll_directory()


if __name__ == "__main__":
    unittest.main()
