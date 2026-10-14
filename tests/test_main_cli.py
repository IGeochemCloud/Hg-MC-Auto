import unittest
import asyncio
from pathlib import Path

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
        self.assertIn("qc", command_names)
        self.assertIn("monitor", command_names)

    def test_fe_workflow_is_registered(self):
        workflow = resolve_workflow_for_element("fe")
        self.assertIn("export", workflow["tasks"])
        self.assertIn("calculate", workflow["tasks"])
        self.assertIn("predict", workflow["tasks"])

    def test_mcp_server_exposes_agent_tools(self):
        self.assertIn("run_element_task", server.available_tools())
        self.assertIn("run_agent_plan", server.available_tools())
        self.assertIn("create_qc_packet", server.available_tools())
        self.assertIn("cd", server.list_elements())

    def test_dispatch_workflow_accepts_fe_and_cd(self):
        self.assertEqual(resolve_workflow_for_element("fe")["element"], "fe")
        self.assertEqual(resolve_workflow_for_element("cd")["element"], "cd")
        self.assertEqual(resolve_workflow_for_element("hg")["support_status"], "production")
        self.assertEqual(resolve_workflow_for_element("fe")["support_status"], "scaffold")

    def test_mcp_server_exposes_release_tools(self):
        self.assertIn("list_platforms", server.available_tools())
        self.assertIn("project_summary", server.available_tools())
        self.assertIn("windows", server.list_platforms())

    def test_bare_command_starts_user_menu(self):
        result = runner.invoke(app, input="0\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Launching Hg-MC-Auto interactive menu", result.output)

    def test_guide_is_honest_about_scaffold_elements(self):
        result = runner.invoke(app, ["guide", "--element", "fe", "--language", "zh"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("实验性脚手架", result.output)

    def test_guide_explains_hg_workflow(self):
        result = runner.invoke(app, ["guide", "--element", "hg", "--language", "zh"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hg 常规流程", result.output)

    def test_stdio_mcp_handshake_lists_and_calls_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root = Path(__file__).resolve().parents[1]
        parameters = StdioServerParameters(
            command=str(root / "hg-mc-auto/bin/hgmc"),
            args=["mcp", "--action", "stdio"],
            cwd=root,
        )

        async def exercise_server():
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()
                    listed_names = {tool.name for tool in tools.tools}
                    result = await session.call_tool("list_elements", {})
                    return initialized.server_info.name, listed_names, result

        server_name, tool_names, result = asyncio.run(exercise_server())
        self.assertEqual(server_name, "hg-mc-auto")
        self.assertIn("list_elements", tool_names)
        self.assertIn("run_agent_plan", tool_names)
        self.assertIn("create_qc_packet", tool_names)
        returned_elements = {item.text for item in result.content if hasattr(item, "text")}
        self.assertIn("hg", returned_elements)


if __name__ == "__main__":
    unittest.main()
