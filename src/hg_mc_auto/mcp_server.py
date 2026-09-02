from __future__ import annotations

import asyncio
import platform
from typing import Any, Dict, List

from hg_mc_auto.core import REGISTRY
from hg_mc_auto.core.workflow import resolve_element_task, resolve_workflow_for_element


class MCPServer:
    """Agent-facing tool surface for workflow selection and execution metadata."""

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
            "version": "0.1.0",
            "elements": self.list_elements(),
            "platforms": self.list_platforms(),
            "current_platform": platform.system().lower(),
            "bilingual": True,
            "mcp_ready": True,
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
            "status": "ok",
            "element": workflow["element"],
            "display_name": workflow["display_name"],
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
            "status": "ok",
            "element": workflow["element"],
            "language": language,
            "plan": resolved_tasks,
        }


server = MCPServer()


try:
    from mcp.server import MCPServer as MCPServerImpl
except ImportError:
    def run_stdio_server() -> None:
        raise RuntimeError("MCP stdio server requires the optional dependency 'mcp'. Install it with: pip install 'hg-mc-auto[mcp]' ")
else:
    mcp_app = MCPServerImpl("hg-mc-auto", version="0.1.0", description="Hg-MC-Auto MCP integration")

    @mcp_app.tool()
    def list_elements_tool() -> list[str]:
        return server.list_elements()

    @mcp_app.tool()
    def get_element_info_tool(name: str) -> dict[str, Any]:
        return server.get_element_info(name)

    @mcp_app.tool()
    def resolve_workflow_tool(element: str) -> dict[str, Any]:
        return server.resolve_workflow(element)

    @mcp_app.tool()
    def run_element_task_tool(element: str, task: str, language: str = "en") -> dict[str, Any]:
        return server.run_element_task(element, task, language=language)

    @mcp_app.tool()
    def project_summary_tool() -> dict[str, Any]:
        return server.project_summary()

    def run_stdio_server() -> None:
        asyncio.run(mcp_app.run_stdio_async())
