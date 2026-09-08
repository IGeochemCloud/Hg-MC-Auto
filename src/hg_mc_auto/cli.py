from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import typer

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hg_mc_auto.core import REGISTRY, build_workflow_context
from hg_mc_auto.core.workflow import resolve_element_task, resolve_workflow_for_element

app = typer.Typer(
    name="hgmc",
    help="General-purpose isotope analysis toolkit with extensible element support.",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main_callback(
    ctx: typer.Context,
) -> None:
    """Launch the interactive workflow when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        from src.main import main as legacy_main
    except ImportError:
        typer.echo("Welcome to Hg-MC-Auto. Use a command such as 'hgmc --help'.")
        return

    typer.echo("Launching Hg-MC-Auto interactive menu...")
    legacy_main()


@app.command("list-elements")
def list_elements() -> None:
    """List supported elements."""
    typer.echo("Supported elements:")
    for name in REGISTRY.list():
        spec = REGISTRY.get(name)
        typer.echo(f"- {name}: {spec.display_name if spec else name}")


@app.command("show-element")
def show_element(name: str = typer.Argument(..., help="Element name such as hg, fe, cd.")) -> None:
    """Show metadata for a specific element."""
    try:
        info = REGISTRY.info(name)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    for key, value in info.items():
        typer.echo(f"{key}: {value}")


@app.command("context")
def make_context(
    element: str = typer.Option("hg", "--element", help="Element to use for workflow context."),
    language: str = typer.Option("en", "--language", help="Language code such as en or zh."),
) -> None:
    """Build a workflow context for an element."""
    context = build_workflow_context(element=element, language=language)
    typer.echo(f"element={context.element}")
    typer.echo(f"language={context.language}")


@app.command("workflow")
def workflow_command(
    element: str = typer.Option("hg", "--element", help="Element whose workflow metadata should be shown."),
) -> None:
    """Display the supported tasks and default workflow for a registered element."""
    try:
        workflow = resolve_workflow_for_element(element)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"element={workflow['element']}")
    typer.echo(f"display_name={workflow['display_name']}")
    typer.echo(f"description={workflow['description']}")
    typer.echo(f"tasks={workflow['tasks']}")
    typer.echo(f"default_task={workflow['default_task']}")
    typer.echo(f"extensions={workflow['extensions']}")


@app.command("run")
def run_command(
    element: str = typer.Option("hg", "--element", "-e", help="Element to execute, such as hg, fe, or cd."),
    task: str = typer.Option("export", "--task", "-t", help="Element task to dispatch."),
    language: str = typer.Option("en", "--language", "-l", help="UI language code such as en or zh."),
    data_path: str | None = typer.Option(None, "--data-path", help="Optional data path for the selected task."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Optional output directory for result files."),
) -> None:
    """Dispatch a supported task for a registered element."""
    try:
        workflow = resolve_workflow_for_element(element)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if task not in workflow["tasks"]:
        typer.echo(f"Task '{task}' is not supported for element '{element}'. Supported tasks: {workflow['tasks']}")
        raise typer.Exit(code=1)

    routed = resolve_element_task(element, task)
    typer.echo(f"Element: {workflow['display_name']}")
    typer.echo(f"Task: {task}")
    typer.echo(f"Script: {routed['script']}")
    typer.echo(f"Language: {language}")
    if data_path:
        typer.echo(f"Data path: {data_path}")
    if output_dir:
        typer.echo(f"Output directory: {output_dir}")
    typer.echo("Dispatch ready for execution.")


@app.command("language")
def language_command(
    lang: str = typer.Option("en", "--lang", "-l", help="Set the language. Supported values: zh, en."),
) -> None:
    """Set the language preference for the CLI workflow."""
    normalized = lang.lower()
    if normalized in {"zh", "cn", "zh-cn", "中文"}:
        typer.echo("Language set to: zh")
        return
    if normalized in {"en", "eng", "english"}:
        typer.echo("Language set to: en")
        return
    typer.echo("Unsupported language. Use 'zh' or 'en'.")
    raise typer.Exit(code=1)


@app.command("platform")
def platform_support() -> None:
    """Report supported platforms and the current environment."""
    current = platform.system()
    supported = {
        "Windows": current == "Windows",
        "Linux": current == "Linux",
        "macOS": current == "Darwin",
    }
    typer.echo("Supported platforms: Windows, Linux, macOS")
    typer.echo(f"Current platform: {current or 'unknown'}")
    for name, is_current in supported.items():
        status = "current" if is_current else "supported"
        typer.echo(f"- {name}: {status}")


@app.command("project-summary")
def project_summary() -> None:
    """Summarize the current project state for agent and release use."""
    typer.echo("name: hg-mc-auto")
    typer.echo("version: 0.1.0")
    typer.echo("elements: hg, fe, cd")
    typer.echo("platforms: windows, linux, macos")
    typer.echo("bilingual: true")
    typer.echo("mcp_ready: true")


@app.command("qc")
def quality_control_command(
    input_path: Path = typer.Option(..., "--input", "-i", exists=True, readable=True, help="JSON file with element, run_id, metrics, results, and flags."),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
    request_llm: bool = typer.Option(False, "--llm", help="Request advisory LLM interpretation after packet creation."),
    language: str = typer.Option("en", "--language", "-l", help="Assessment language: en or zh."),
    model: str = typer.Option("orcarouter/auto", "--model", help="OpenAI-compatible model identifier."),
) -> None:
    """Create a structured QC packet and optionally request advisory interpretation."""
    from hg_mc_auto.quality_control import packet_from_result_file, request_llm_assessment

    try:
        packet = packet_from_result_file(input_path)
        payload = packet.to_dict()
        if request_llm:
            payload["llm_assessment"] = request_llm_assessment(
                packet,
                model=model,
                language=language,
            )
    except (KeyError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"QC processing failed: {exc}", err=True)
        raise typer.Exit(code=1)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output_path.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"QC output written to {output_path}")
    else:
        typer.echo(rendered)


@app.command("monitor")
def monitor_command(
    directory: Path = typer.Option(..., "--directory", "-d", exists=True, file_okay=False, readable=True),
    element: str = typer.Option("hg", "--element", "-e"),
    interval: float = typer.Option(2.0, "--interval", min=0.1),
    once: bool = typer.Option(False, "--once", help="Scan once and exit."),
) -> None:
    """Monitor a result directory and emit structured QC packets as JSON."""
    from hg_mc_auto.monitor import monitor_results

    try:
        for packet in monitor_results(directory, element=element, interval=interval, once=once):
            typer.echo(packet.to_json())
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Monitoring failed: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command("mcp")
def mcp_command(
    action: str = typer.Option("status", "--action", "-a", help="Action: status, tools, json, stdio."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Expose the agent-facing MCP tool surface for direct use or stdio hosting."""
    from hg_mc_auto.mcp_server import server

    if action == "status":
        payload = {
            "name": "hg-mc-auto",
            "mcp_ready": True,
            "tools": server.available_tools(),
            "elements": server.list_elements(),
        }
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            typer.echo("MCP status: ready")
            typer.echo(f"Tools: {', '.join(payload['tools'])}")
            typer.echo(f"Elements: {', '.join(payload['elements'])}")
        return

    if action == "tools":
        tools = server.available_tools()
        if json_output:
            typer.echo(json.dumps({"tools": tools}, ensure_ascii=False, indent=2))
        else:
            for tool in tools:
                typer.echo(tool)
        return

    if action == "json":
        typer.echo(json.dumps({
            "elements": server.list_elements(),
            "tools": server.available_tools(),
            "summary": server.project_summary(),
        }, ensure_ascii=False, indent=2))
        return

    if action == "stdio":
        try:
            from hg_mc_auto.mcp_server import run_stdio_server
        except ImportError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        run_stdio_server()
        return

    typer.echo("Unsupported MCP action. Use one of: status, tools, json, stdio.", err=True)
    raise typer.Exit(code=1)


@app.command("release")
def release_command(
    action: str = typer.Option("help", "--action", "-a", help="Action: check, build, publish, sync."),
    version: str | None = typer.Option(None, "--version", help="Version tag to publish, e.g. 0.1.1."),
    message: str = typer.Option("chore: release", "--message", help="Git commit message used by the sync flow."),
) -> None:
    """Provide a release workflow for PyPI and GitHub synchronization."""
    from hg_mc_auto.release import app as release_app

    if action == "help":
        typer.echo("Release actions: check, build, publish, sync")
        return

    if action == "check":
        release_app("check")
        return

    if action == "build":
        release_app("build")
        return

    if action == "publish":
        release_app("publish")
        return

    if action == "sync":
        if version is None:
            typer.echo("A version tag is required for sync, e.g. --version 0.1.1", err=True)
            raise typer.Exit(code=1)
        release_app(["sync", "--version", version, "--message", message])
        return

    typer.echo("Unsupported release action. Use one of: check, build, publish, sync.", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
