import unittest

from typer.testing import CliRunner

from src.main import app as legacy_app
from hg_mc_auto.cli import app
from hg_mc_auto.core.workflow import resolve_workflow_for_element
from hg_mc_auto.mcp_server import server


runner = CliRunner()


class TestMainCli(unittest.TestCase):
    def test_app_exposes_element_registry_commands(self):
        command_names = {command.name for command in app.registered_commands}
        self.assertIn("list-elements", command_names)
        self.assertIn("show-element", command_names)
        self.assertIn("context", command_names)
        self.assertIn("mcp", command_names)
        self.assertIn("release", command_names)

    def test_fe_workflow_is_registered(self):
        workflow = resolve_workflow_for_element("fe")
        self.assertIn("export", workflow["tasks"])
        self.assertIn("calculate", workflow["tasks"])
        self.assertIn("predict", workflow["tasks"])

    def test_mcp_server_exposes_agent_tools(self):
        self.assertIn("run_element_task", server.available_tools())
        self.assertIn("run_agent_plan", server.available_tools())
        self.assertIn("cd", server.list_elements())

    def test_dispatch_workflow_accepts_fe_and_cd(self):
        self.assertEqual(resolve_workflow_for_element("fe")["element"], "fe")
        self.assertEqual(resolve_workflow_for_element("cd")["element"], "cd")

    def test_mcp_server_exposes_release_tools(self):
        self.assertIn("list_platforms", server.available_tools())
        self.assertIn("project_summary", server.available_tools())
        self.assertIn("windows", server.list_platforms())

    def test_bare_command_starts_user_menu(self):
        result = runner.invoke(app, input="0\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Launching Hg-MC-Auto interactive menu", result.output)


if __name__ == "__main__":
    unittest.main()
