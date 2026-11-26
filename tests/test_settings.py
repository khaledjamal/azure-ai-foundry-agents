# tests/test_settings.py
"""Tests for workshop_utils.settings: every lab script loads the repo's `.env` with
load_env().

- A missing `.env` stops the script at once, with a hint to copy `.env.example`, so
  every script reacts the same way instead of failing later on one missing setting.
- An existing `.env` is loaded into os.environ, and its path is returned.

Run from the repo root:
    uv run python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workshop_utils import settings

TEST_VARIABLE = "WORKSHOP_SETTINGS_TEST_VALUE"


class LoadEnvTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.env_file = Path(temp_dir.name) / ".env"

        env_file_patch = mock.patch.object(settings, "ENV_FILE", self.env_file)
        env_file_patch.start()
        self.addCleanup(env_file_patch.stop)

        self.addCleanup(os.environ.pop, TEST_VARIABLE, None)

    def test_missing_env_file_is_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, r"copy \.env\.example"):
            settings.load_env()

    def test_existing_env_file_is_loaded(self):
        self.env_file.write_text(f"{TEST_VARIABLE}=loaded\n", encoding="utf-8")

        returned_path = settings.load_env()

        self.assertEqual(returned_path, self.env_file)
        self.assertEqual(os.environ.get(TEST_VARIABLE), "loaded")


if __name__ == "__main__":
    unittest.main()
