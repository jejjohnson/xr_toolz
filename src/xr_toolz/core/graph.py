"""Functional Graph API — Input, Node, and Graph.

Mirrors the Keras functional API: build a computation DAG by calling
operators on symbolic :class:`Input` nodes, then compile the graph into
a :class:`Graph` operator. The DAG is executed eagerly in topological
order when the graph is called with concrete data.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from xr_toolz.core.operator import Operator


class Node:
    """A symbolic placeholder for an intermediate result in a Graph.

    Nodes are created automatically when operators are called on
    :class:`Input` instances or other ``Node`` instances. Users do not
    instantiate these directly.
    """

    def __init__(
        self,
        operator: Operator | None,
        parents: tuple[Node, ...],
        name: str | None = None,
    ):
        self.operator = operator
        self.parents = tuple(parents)
        self.name = name

    def __repr__(self) -> str:
        if self.operator is None:
            return f"Input(name={self.name!r})"
        return f"Node({self.operator!r}, parents={len(self.parents)})"


class Input(Node):
    """A named entry point into a computation graph.

    Inputs have no parents and no operator. They are pure placeholders
    bound to real datasets at execution time.
    """

    def __init__(self, name: str):
        super().__init__(operator=None, parents=(), name=name)


class Graph:
    """A computation DAG compiled from symbolic :class:`Node` connections.

    ``Graph`` is not itself an :class:`~xr_toolz.core.operator.Operator`
    in v0.1 — nesting graphs inside graphs is deferred. Graphs do compose
    inside a :class:`~xr_toolz.core.sequential.Sequential` as long as
    they are single-input / single-output, because Sequential invokes
    each step positionally.
    """

    def __init__(
        self,
        inputs: dict[str, Input],
        outputs: dict[str, Node],
    ):
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        self._execution_order = _topological_sort(self.inputs, self.outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bindings = self._bind(args, kwargs)
        results = self._execute(bindings)
        if args and not kwargs and len(self.outputs) == 1:
            (only_output,) = self.outputs.values()
            return results[id(only_output)]
        return {name: results[id(node)] for name, node in self.outputs.items()}

    def _bind(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[int, Any]:
        if args and not kwargs:
            if len(args) != 1 or len(self.inputs) != 1:
                raise ValueError(
                    "Positional call requires exactly one input and one "
                    f"argument; got {len(args)} args and "
                    f"{len(self.inputs)} inputs."
                )
            (only_input,) = self.inputs.values()
            return {id(only_input): args[0]}

        missing = set(self.inputs) - set(kwargs)
        extra = set(kwargs) - set(self.inputs)
        if missing or extra:
            raise ValueError(
                f"Graph inputs mismatch. missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        return {id(node): kwargs[name] for name, node in self.inputs.items()}

    def _execute(self, bindings: dict[int, Any]) -> dict[int, Any]:
        cache: dict[int, Any] = dict(bindings)
        for node in self._execution_order:
            if id(node) in cache:
                continue
            if node.operator is None:
                raise ValueError(f"Input node {node.name!r} was not bound to data.")
            parent_values = tuple(cache[id(p)] for p in node.parents)
            cache[id(node)] = node.operator._apply(*parent_values)
        return cache

    def describe(self) -> str:
        """Pretty-print the graph structure."""
        lines = [
            f"Graph ({len(self.inputs)} inputs, {len(self.outputs)} outputs):",
            f"  Inputs: {list(self.inputs.keys())}",
        ]
        for i, node in enumerate(self._execution_order):
            if node.operator is None:
                continue
            parent_labels = [p.name or f"<node {id(p)}>" for p in node.parents]
            lines.append(f"  [{i}] {node.operator!r} ← {parent_labels}")
        lines.append(f"  Outputs: {list(self.outputs.keys())}")
        return "\n".join(lines)


def _topological_sort(
    inputs: dict[str, Input],
    outputs: dict[str, Node],
) -> list[Node]:
    """Kahn's algorithm over the reachable subgraph rooted at ``outputs``."""

    reachable: dict[int, Node] = {}
    stack: list[Node] = list(outputs.values())
    while stack:
        node = stack.pop()
        if id(node) in reachable:
            continue
        reachable[id(node)] = node
        stack.extend(node.parents)

    input_ids = {id(node) for node in inputs.values()}
    for node_id in input_ids:
        if node_id not in reachable:
            raise ValueError("At least one declared Input is not used by any output.")

    in_degree: dict[int, int] = {
        nid: sum(1 for p in node.parents if id(p) in reachable)
        for nid, node in reachable.items()
    }
    children: dict[int, list[Node]] = {nid: [] for nid in reachable}
    for node in reachable.values():
        for parent in node.parents:
            if id(parent) in reachable:
                children[id(parent)].append(node)

    queue: deque[Node] = deque(
        reachable[nid] for nid, deg in in_degree.items() if deg == 0
    )
    ordered: list[Node] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in children[id(node)]:
            in_degree[id(child)] -= 1
            if in_degree[id(child)] == 0:
                queue.append(child)

    if len(ordered) != len(reachable):
        raise ValueError("Graph contains a cycle.")
    return ordered
