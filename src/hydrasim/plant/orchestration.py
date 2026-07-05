"""Live orchestration planning and launcher for staged HydraSim profiles."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Sequence

from .models import PlantNode
from .runtime import build_runtime_artifact


@dataclass(frozen=True)
class LiveLaunchNode:
    node_id: str
    label: str
    area: str
    stage: str
    host: str
    port: int
    command: tuple[str, ...]
    log_path: Path | None


@dataclass(frozen=True)
class LiveOrchestrationPlan:
    profile_id: str
    scenario_id: str
    area: str
    stage: str
    duration_seconds: float
    startup_delay_seconds: float
    nodes: Sequence[LiveLaunchNode]
    limitations: Sequence[str]


@dataclass(frozen=True)
class LiveLaunchResult:
    node_id: str
    port: int
    return_code: int | None
    status: str


def _launchable(node: PlantNode) -> bool:
    return node.modbus_port is not None and node.stage in {
        "field-device",
        "field-controller",
    }


def build_live_orchestration_plan(
    profile_id: str,
    scenario_id: str,
    area: str,
    stage: str,
    host: str = "127.0.0.1",
    base_port: int = 5520,
    duration_seconds: float = 30.0,
    startup_delay_seconds: float = 1.0,
    log_dir: str | Path | None = None,
) -> LiveOrchestrationPlan:
    if stage == "offline-export":
        raise ValueError("live orchestration requires a non-offline stage")
    if area == "all":
        raise ValueError("live orchestration requires a selected area")
    if not 1 <= base_port <= 65535:
        raise ValueError("base port must be in TCP range")
    if duration_seconds <= 0.0:
        raise ValueError("duration must be positive")

    artifact = build_runtime_artifact(profile_id, scenario_id, area, stage)
    nodes = tuple(node for node in artifact.active_nodes if _launchable(node))
    if not nodes:
        raise ValueError("selected stage has no launchable Modbus endpoints")
    if base_port + len(nodes) - 1 > 65535:
        raise ValueError("base port range exceeds TCP range")

    logs = Path(log_dir) if log_dir is not None else None
    launch_nodes: list[LiveLaunchNode] = []
    for offset, node in enumerate(sorted(nodes, key=lambda item: item.node_id)):
        port = base_port + offset
        command = (
            sys.executable,
            "-m",
            "hydrasim",
            "--host",
            host,
            "--port",
            str(port),
            "--duration",
            str(duration_seconds),
        )
        log_path = logs / f"{node.node_id}.log" if logs is not None else None
        launch_nodes.append(
            LiveLaunchNode(
                node.node_id,
                node.label,
                node.area,
                node.stage,
                host,
                port,
                command,
                log_path,
            )
        )

    return LiveOrchestrationPlan(
        profile_id,
        scenario_id,
        area,
        stage,
        duration_seconds,
        startup_delay_seconds,
        tuple(launch_nodes),
        tuple(artifact.limitations)
        + (
            "Live orchestration launches synthetic HydraSim endpoints only.",
            "Each launched endpoint uses the common water-process runtime.",
        ),
    )


def render_live_plan(plan: LiveOrchestrationPlan) -> str:
    rows = [
        "node_id,area,stage,host,port,command",
    ]
    for node in plan.nodes:
        rows.append(
            ",".join(
                (
                    node.node_id,
                    node.area,
                    node.stage,
                    node.host,
                    str(node.port),
                    " ".join(node.command),
                )
            )
        )
    return "\n".join(rows) + "\n"


def run_live_orchestration(
    plan: LiveOrchestrationPlan,
    process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[LiveLaunchResult, ...]:
    processes: list[tuple[LiveLaunchNode, subprocess.Popen, BinaryIO | None]] = []
    try:
        for node in plan.nodes:
            log_handle: BinaryIO | None
            if node.log_path is not None:
                node.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = node.log_path.open("wb")
            else:
                log_handle = None
            process = process_factory(
                node.command,
                stdout=log_handle if log_handle is not None else subprocess.DEVNULL,
                stderr=log_handle if log_handle is not None else subprocess.DEVNULL,
            )
            processes.append((node, process, log_handle))
            if plan.startup_delay_seconds > 0.0:
                sleeper(plan.startup_delay_seconds)

        sleeper(plan.duration_seconds)
    finally:
        for _node, process, _log_handle in processes:
            if process.poll() is None:
                process.terminate()
        for _node, process, log_handle in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            if log_handle is not None:
                log_handle.close()

    return tuple(
        LiveLaunchResult(
            node.node_id,
            node.port,
            process.poll(),
            "completed" if process.poll() == 0 else "stopped",
        )
        for node, process, _log_handle in processes
    )
