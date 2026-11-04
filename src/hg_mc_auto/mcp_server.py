from __future__ import annotations

import asyncio
import platform
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

from hg_mc_auto import __version__
from hg_mc_auto.core import REGISTRY
from hg_mc_auto.core.workflow import resolve_element_task, resolve_workflow_for_element


class MCPServer:
    """Agent-facing metadata, routing, and quality-control tools."""

    def __init__(self) -> None:
        self._seen_result_files: Dict[str, Dict[str, int]] = {}
        self._scan_lock = threading.Lock()

    def list_elements(self) -> List[str]:
        return REGISTRY.list()

    def get_element_info(self, name: str) -> Dict[str, Any]:
        return REGISTRY.info(name)

    def resolve_workflow(self, element: str) -> Dict[str, Any]:
        return resolve_workflow_for_element(element)

    def list_platforms(self) -> List[str]:
        return ["windows", "linux", "macos"]

    def project_summary(self) -> Dict[str, Any]:
        return {
            "name": "hg-mc-auto",
            "version": __version__,
            "elements": self.list_elements(),
            "platforms": self.list_platforms(),
            "current_platform": platform.system().lower(),
            "bilingual": True,
            "mcp_ready": MCP_AVAILABLE,
            "scientific_task_execution": False,
        }

    def available_tools(self) -> List[str]:
        return [
            "list_elements",
            "get_element_info",
            "resolve_workflow",
            "run_element_task",
            "run_agent_plan",
            "list_platforms",
            "project_summary",
            "create_qc_packet",
            "create_qc_packets",
            "assess_qc_file",
            "scan_result_directory",
        ]

    def run_element_task(
        self,
        element: str,
        task: str,
        *,
        data_path: str | None = None,
        output_dir: str | None = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        workflow = resolve_workflow_for_element(element)
        if task not in workflow["tasks"]:
            raise ValueError(f"Task '{task}' is not supported for element '{element}'.")

        routed = resolve_element_task(element, task)
        return {
            "status": "not_executed",
            "executed": False,
            "message": "This tool resolves workflow metadata only; it does not execute scientific tasks.",
            "element": workflow["element"],
            "display_name": workflow["display_name"],
            "support_status": workflow["support_status"],
            "task": task,
            "script": routed["script"],
            "language": language,
            "data_path": data_path,
            "output_dir": output_dir,
            "supported_tasks": workflow["tasks"],
        }

    def run_agent_plan(self, element: str, tasks: List[str], *, language: str = "en") -> Dict[str, Any]:
        workflow = resolve_workflow_for_element(element)
        resolved_tasks = []
        for task in tasks:
            if task not in workflow["tasks"]:
                raise ValueError(f"Task '{task}' is not supported for element '{element}'.")
            resolved_tasks.append(resolve_element_task(element, task))

        return {
            "status": "plan_only",
            "executed": False,
            "message": "This tool creates a task plan only; it does not execute scientific tasks.",
            "element": workflow["element"],
            "support_status": workflow["support_status"],
            "language": language,
            "plan": resolved_tasks,
        }

    def assess_qc_file(
        self,
        path: str,
        *,
        element: str = "hg",
        model: str = "orcarouter/auto",
        language: str = "en",
    ) -> Dict[str, Any]:
        from hg_mc_auto.quality_control import packet_from_result_file, request_llm_assessment

        packet = packet_from_result_file(path, element=element)
        assessment = request_llm_assessment(packet, model=model, language=language)
        return {
            "packet": packet.to_dict(),
            "llm_assessment": assessment,
        }

    def scan_result_directory(self, directory: str, *, element: str = "hg") -> Dict[str, Any]:
        from hg_mc_auto.export_monitor import wait_for_stable_file
        from hg_mc_auto.monitor import changed_result_files
        from hg_mc_auto.quality_control import packets_from_result_file

        result_directory = Path(directory)
        if not result_directory.is_dir():
            raise NotADirectoryError(f"Result directory does not exist: {result_directory}")

        with self._scan_lock:
            seen = self._seen_result_files.setdefault(str(result_directory.resolve()), {})
            changed_paths = changed_result_files(result_directory, seen)
            packets = []
            scanned_files = []
            pending_files = []
            for path in changed_paths:
                if not wait_for_stable_file(path, timeout=10.0):
                    seen.pop(str(path.resolve()), None)
                    pending_files.append(str(path))
                    continue
                packets.extend(packets_from_result_file(path, element=element))
                scanned_files.append(str(path))

        return {
            "element": element.lower(),
            "scanned_files": scanned_files,
            "pending_files": pending_files,
            "count": len(packets),
            "packets": [packet.to_dict() for packet in packets],
        }


server = MCPServer()


try:
    from mcp.server import MCPServer as MCPServerImpl
except ImportError:
    MCP_AVAILABLE = False

    def run_stdio_server() -> None:
        raise RuntimeError(
            "MCP stdio requires Python 3.10+ and the optional dependency 'mcp'. "
            "Install it with: pip install 'hg-mc-auto[mcp]'"
        )
else:
    from mcp.server.mcpserver.exceptions import ToolError

    MCP_AVAILABLE = True
    mcp_app = MCPServerImpl("hg-mc-auto", version=__version__, description="Hg-MC-Auto MCP integration")

    def call_tool(operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ToolError(str(message)) from exc

    @mcp_app.tool(name="list_elements", description="List registered isotope elements and their support maturity.")
    def list_elements_tool() -> list[str]:
        return server.list_elements()

    @mcp_app.tool(name="get_element_info", description="Get metadata, supported tasks, and maturity for an element.")
    def get_element_info_tool(name: str) -> dict[str, Any]:
        return call_tool(lambda: server.get_element_info(name))

    @mcp_app.tool(name="resolve_workflow", description="Resolve an element workflow and its supported tasks.")
    def resolve_workflow_tool(element: str) -> dict[str, Any]:
        return call_tool(lambda: server.resolve_workflow(element))

    @mcp_app.tool(name="run_element_task", description="Resolve a task route. This currently reports the target script; it does not launch the scientific workflow.")
    def run_element_task_tool(element: str, task: str, language: str = "en") -> dict[str, Any]:
        return call_tool(lambda: server.run_element_task(element, task, language=language))

    @mcp_app.tool(name="run_agent_plan", description="Build an ordered task plan for a registered element; does not execute instrument operations.")
    def run_agent_plan_tool(element: str, tasks: list[str], language: str = "en") -> dict[str, Any]:
        return call_tool(lambda: server.run_agent_plan(element, tasks, language=language))

    @mcp_app.tool(name="list_platforms", description="List supported operating systems.")
    def list_platforms_tool() -> list[str]:
        return server.list_platforms()

    @mcp_app.tool(name="project_summary", description="Return project version, supported elements, and platform summary.")
    def project_summary_tool() -> dict[str, Any]:
        return server.project_summary()

    @mcp_app.tool(name="create_qc_packet", description="Create a structured, human-review-required QC packet from a JSON, CSV, or Excel result file.")
    def create_qc_packet_tool(path: str, element: str = "hg") -> dict[str, Any]:
        from hg_mc_auto.quality_control import packet_from_result_file

        return call_tool(lambda: packet_from_result_file(path, element=element).to_dict())

    @mcp_app.tool(name="create_qc_packets", description="Create one human-review-required QC packet for every row in a JSON, CSV, or Excel result file.")
    def create_qc_packets_tool(path: str, element: str = "hg") -> dict[str, Any]:
        from hg_mc_auto.quality_control import packets_from_result_file

        packets = call_tool(lambda: packets_from_result_file(path, element=element))
        return {
            "element": element.lower(),
            "count": len(packets),
            "packets": [packet.to_dict() for packet in packets],
        }

    @mcp_app.tool(name="assess_qc_file", description="Request advisory LLM interpretation of a single result row. Only the structured QC packet is sent to the configured provider; set ORCA_KEY first.")
    def assess_qc_file_tool(
        path: str,
        element: str = "hg",
        model: str = "orcarouter/auto",
        language: str = "en",
    ) -> dict[str, Any]:
        return call_tool(
            lambda: server.assess_qc_file(
                path,
                element=element,
                model=model,
                language=language,
            )
        )

    @mcp_app.tool(name="scan_result_directory", description="Scan a result directory once for new or modified stable JSON, CSV, or Excel files and create review-required packets for each row.")
    def scan_result_directory_tool(directory: str, element: str = "hg") -> dict[str, Any]:
        return call_tool(lambda: server.scan_result_directory(directory, element=element))

    def run_stdio_server() -> None:
        asyncio.run(mcp_app.run_stdio_async())
