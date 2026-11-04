from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .registry import REGISTRY


@dataclass
class WorkflowContext:
    element: str
    language: str = "en"
    root_dir: Optional[str] = None
    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


WORKFLOW_LIBRARY = {
    "hg": {
        "label": "Hg",
        "tasks": ["export", "calculate", "empirical", "predict", "train"],
        "default_task": "export",
        "description": "Mercury isotope processing and QC workflow.",
    },
    "fe": {
        "label": "Fe",
        "tasks": ["export", "calculate", "predict"],
        "default_task": "export",
        "description": "Iron isotope processing and prediction workflow.",
    },
    "cd": {
        "label": "Cd",
        "tasks": ["export", "calculate", "predict"],
        "default_task": "export",
        "description": "Cadmium isotope processing and prediction workflow.",
    },
}


def resolve_workflow_for_element(element: str) -> Dict[str, Any]:
    spec = REGISTRY.get(element)
    if spec is None:
        raise KeyError(f"Unsupported element: {element}")

    base = WORKFLOW_LIBRARY.get(spec.name.lower(), {
        "label": spec.display_name,
        "tasks": spec.supported_tasks,
        "default_task": spec.supported_tasks[0] if spec.supported_tasks else "export",
        "description": spec.description,
    })

    return {
        "element": spec.name,
        "display_name": spec.display_name,
        "tasks": base["tasks"],
        "default_task": base["default_task"],
        "description": base["description"],
        "extensions": list(spec.default_extensions),
        "support_status": spec.support_status,
    }


def resolve_element_task(element: str, task: str) -> Dict[str, Any]:
    workflow = resolve_workflow_for_element(element)
    if task not in workflow["tasks"]:
        raise ValueError(f"Task '{task}' is not supported for element '{element}'.")

    if workflow["support_status"] != "production":
        return {
            "element": workflow["element"],
            "task": task,
            "script": None,
            "implementation_status": "scaffold",
            "description": (
                f"{workflow['display_name']} is a scaffold; "
                "no element-specific task script is implemented."
            ),
        }

    task_map = {
        "export": "1_Automatic_export.py",
        "calculate": "2_Automatic_calculation.py",
        "empirical": "3_Empirical_model.py",
        "predict": "4_ML_Predict.py",
        "train-external": "5.Exter_ML_train.py",
        "train-internal": "6.Inter_ML_train.py",
    }
    script = task_map.get(task)

    return {
        "element": workflow["element"],
        "task": task,
        "script": script,
        "scripts": (
            ["5.Exter_ML_train.py", "6.Inter_ML_train.py"]
            if task == "train"
            else ([script] if script else [])
        ),
        "implementation_status": "available" if script else "requires_task_selection",
        "description": (
            "Select external or internal training before execution."
            if task == "train"
            else f"Run the {task} workflow for {workflow['display_name']}"
        ),
    }


def build_workflow_context(
    element: str,
    *,
    language: str = "en",
    root_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowContext:
    resolved = resolve_workflow_for_element(element)
    return WorkflowContext(
        element=resolved["element"],
        language=language.lower(),
        root_dir=root_dir,
        output_dir=output_dir,
        metadata={"workflow": resolved, **(metadata or {})},
    )
