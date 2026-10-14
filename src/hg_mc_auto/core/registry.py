from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class ElementSpec:
    name: str
    display_name: str
    description: str
    default_workflow: str = "default"
    supported_tasks: List[str] = field(default_factory=list)
    default_extensions: tuple[str, ...] = (".dat", ".log", ".csv", ".xlsx")
    support_status: str = "planned"
    metadata: Dict[str, object] = field(default_factory=dict)


class ElementRegistry:
    """Register supported isotopic elements and their workflow metadata."""

    def __init__(self) -> None:
        self._elements: Dict[str, ElementSpec] = {}

    def register(self, spec: ElementSpec) -> None:
        self._elements[spec.name.lower()] = spec

    def register_many(self, specs: Iterable[ElementSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> Optional[ElementSpec]:
        return self._elements.get(name.lower())

    def list(self) -> List[str]:
        return sorted(self._elements.keys())

    def info(self, name: str) -> Dict[str, object]:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unsupported element: {name}")
        return {
            "name": spec.name,
            "display_name": spec.display_name,
            "description": spec.description,
            "default_workflow": spec.default_workflow,
            "tasks": spec.supported_tasks,
            "extensions": list(spec.default_extensions),
            "support_status": spec.support_status,
            "metadata": spec.metadata,
        }


REGISTRY = ElementRegistry()
REGISTRY.register_many(
    [
        ElementSpec(
            name="hg",
            display_name="Hg",
            description="Mercury isotope analysis workflow.",
            default_workflow="default",
            supported_tasks=["export", "calculate", "empirical", "predict", "train"],
            default_extensions=(".dat", ".log", ".csv", ".xlsx"),
            support_status="production",
            metadata={"family": "mercury", "legacy": True},
        ),
        ElementSpec(
            name="fe",
            display_name="Fe",
            description="Iron isotope workflow scaffold for future extensions.",
            default_workflow="default",
            supported_tasks=["export", "calculate", "predict"],
            default_extensions=(".dat", ".csv", ".xlsx"),
            support_status="scaffold",
            metadata={"family": "iron", "legacy": False},
        ),
        ElementSpec(
            name="cd",
            display_name="Cd",
            description="Cadmium isotope workflow scaffold for future extensions.",
            default_workflow="default",
            supported_tasks=["export", "calculate", "predict"],
            default_extensions=(".dat", ".csv", ".xlsx"),
            support_status="scaffold",
            metadata={"family": "cadmium", "legacy": False},
        ),
    ]
)


def list_supported_elements() -> List[str]:
    return REGISTRY.list()
