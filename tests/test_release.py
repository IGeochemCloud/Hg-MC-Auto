import unittest
from pathlib import Path

from hg_mc_auto.release import read_project_version


class TestReleaseHelpers(unittest.TestCase):
    def test_read_project_version_matches_pyproject(self):
        root = Path(__file__).resolve().parents[1]
        version = read_project_version()
        self.assertEqual(version, "0.1.0")


if __name__ == "__main__":
    unittest.main()