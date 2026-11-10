# tests/test_file_paths.py
"""Every Python file in the repo starts with a comment holding its own path.

- Why: a file copied into an editor, a note or a chat still says where it came from.
- This test fails when a file is renamed or moved and its first line still names the
  old path.
- Only files git tracks are checked: a new file counts once it is added with git add.

Run from the repo root:
    uv run python -m unittest discover -s tests -v
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def tracked_python_files():
    git_result = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return git_result.stdout.split()


class FilePathCommentTest(unittest.TestCase):
    def test_first_line_is_the_file_path(self):
        python_files = tracked_python_files()
        self.assertTrue(python_files, "git ls-files found no Python files")
        for relative_path in python_files:
            with self.subTest(file=relative_path):
                text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                first_line = text.partition("\n")[0]
                self.assertEqual(first_line, f"# {relative_path}")


if __name__ == "__main__":
    unittest.main()
