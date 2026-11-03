import re
import unittest
from pathlib import Path

from hg_mc_auto.release import read_project_version


class TestReleaseHelpers(unittest.TestCase):
    def test_read_project_version_matches_pyproject(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        expected = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE).group(1)
        version = read_project_version()
        self.assertEqual(version, expected)


if __name__ == "__main__":
    unittest.main()