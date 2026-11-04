import unittest
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from src.main import app as legacy_app
from hg_mc_auto.cli import app
from hg_mc_auto.core.workflow import resolve_element_task, resolve_workflow_for_element
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
        self.assertIn("create_qc_packets", server.available_tools())
        self.assertIn("assess_qc_file", server.available_tools())
        self.assertIn("scan_result_directory", server.available_tools())
        self.assertIn("cd", server.list_elements())

    def test_dispatch_workflow_accepts_fe_and_cd(self):
        self.assertEqual(resolve_workflow_for_element("fe")["element"], "fe")
        self.assertEqual(resolve_workflow_for_element("cd")["element"], "cd")
        self.assertEqual(resolve_workflow_for_element("hg")["support_status"], "production")
        self.assertEqual(resolve_workflow_for_element("fe")["support_status"], "scaffold")
        self.assertIsNone(resolve_element_task("fe", "predict")["script"])
        training_route = resolve_element_task("hg", "train")
        self.assertEqual(training_route["implementation_status"], "requires_task_selection")
        self.assertEqual(len(training_route["scripts"]), 2)

    def test_qc_command_returns_a_packet_for_each_input_row(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.json"
            input_path.write_text(
                json.dumps([
                    {"run_id": "run-1", "Data_Quality_Score": 0.8},
                    {"run_id": "run-2", "Data_Quality_Score": 0.2},
                ]),
                encoding="utf-8",
            )
            result = runner.invoke(app, ["qc", "--input", str(input_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(json.loads(result.output)), 2)

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
        import shutil
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root = Path(__file__).resolve().parents[1]
        # Use system hgmc (installed via pip install -e) instead of local venv path
        hgmc_path = shutil.which("hgmc") or str(root / "hg-mc-auto/bin/hgmc")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "hg_mc_auto.cli", "mcp", "--action", "stdio"],
            cwd=root,
        )

        async def exercise_server():
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()
                    listed_names = {tool.name for tool in tools.tools}
                    result = await session.call_tool("list_elements", {})
                    element_info = await session.call_tool("get_element_info", {"name": "hg"})
                    workflow = await session.call_tool("resolve_workflow", {"element": "hg"})
                    platforms = await session.call_tool("list_platforms", {})
                    summary = await session.call_tool("project_summary", {})
                    route = await session.call_tool(
                        "run_element_task",
                        {"element": "hg", "task": "empirical"},
                    )
                    plan = await session.call_tool(
                        "run_agent_plan",
                        {"element": "hg", "tasks": ["empirical", "predict"]},
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        result_path = Path(directory) / "rows.json"
                        result_path.write_text(
                            json.dumps([
                                {"run_id": "run-1", "Data_Quality_Score": 0.8},
                                {"run_id": "run-2", "Data_Quality_Score": 0.2},
                            ]),
                            encoding="utf-8",
                        )
                        batch = await session.call_tool(
                            "create_qc_packets",
                            {"path": str(result_path), "element": "hg"},
                        )
                        scan = await session.call_tool(
                            "scan_result_directory",
                            {"directory": directory, "element": "hg"},
                        )
                        unchanged_scan = await session.call_tool(
                            "scan_result_directory",
                            {"directory": directory, "element": "hg"},
                        )
                    invalid = await session.call_tool(
                        "get_element_info",
                        {"name": "unknown"},
                    )
                    return (
                        initialized.server_info.name,
                        listed_names,
                        result,
                        route,
                        batch,
                        invalid,
                        scan,
                        unchanged_scan,
                        element_info,
                        workflow,
                        platforms,
                        summary,
                        plan,
                    )

        (
            server_name,
            tool_names,
            result,
            route,
            batch,
            invalid,
            scan,
            unchanged_scan,
            element_info,
            workflow,
            platforms,
            summary,
            plan,
        ) = asyncio.run(exercise_server())
        self.assertEqual(server_name, "hg-mc-auto")
        self.assertIn("list_elements", tool_names)
        self.assertIn("run_agent_plan", tool_names)
        self.assertIn("create_qc_packet", tool_names)
        self.assertIn("create_qc_packets", tool_names)
        self.assertIn("assess_qc_file", tool_names)
        self.assertIn("scan_result_directory", tool_names)
        returned_elements = {item.text for item in result.content if hasattr(item, "text")}
        self.assertIn("hg", returned_elements)
        self.assertEqual(route.structured_content["status"], "not_executed")
        self.assertFalse(route.structured_content["executed"])
        self.assertEqual(batch.structured_content["count"], 2)
        self.assertEqual(scan.structured_content["count"], 2)
        self.assertEqual(unchanged_scan.structured_content["count"], 0)
        self.assertEqual(element_info.structured_content["name"], "hg")
        self.assertEqual(workflow.structured_content["element"], "hg")
        self.assertIn("windows", platforms.structured_content["result"])
        self.assertTrue(summary.structured_content["mcp_ready"])
        self.assertEqual(plan.structured_content["status"], "plan_only")
        self.assertTrue(invalid.is_error)
        self.assertIn("Unsupported element", invalid.content[0].text)


if __name__ == "__main__":
    unittest.main()
