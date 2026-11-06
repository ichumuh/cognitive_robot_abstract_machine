"""
Temporary CI diagnostic: records every QP solve and the world state after every tick of
each StatechartExecutor.tick_until_end call and writes them to $TICK_TRACE_DIR.
"""

import json
import os
import platform
from pathlib import Path

import numpy as np
import piqp
from cramph.executor import StatechartExecutor
from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP

_trace_dir = Path(os.environ.get("TICK_TRACE_DIR", "tick_traces"))
_trace_dir.mkdir(parents=True, exist_ok=True)
_solves: list[dict] = []
_original_solver_call = QPSolverPIQP.solver_call_explicit_interface
_original_tick_until_end = StatechartExecutor.tick_until_end


def _recording_solver_call(self, qp_data):
    try:
        x = _original_solver_call(self, qp_data)
        failed = None
    except Exception as error:
        x = None
        failed = repr(error)
    info = self.solver.result.info
    _solves.append(
        dict(
            status=str(info.status),
            iterations=int(info.iter),
            primal_residual=float(info.primal_res),
            dual_residual=float(info.dual_res),
            duality_gap=float(info.duality_gap),
            number_of_variables=int(len(qp_data.linear_weights)),
            number_of_inequalities=int(len(qp_data.inequality_upper_bounds)),
            solution_norm=None if x is None else float(np.linalg.norm(x)),
            solution=None if x is None else [float(v) for v in x],
            failed=failed,
        )
    )
    if failed is not None:
        raise
    return x


def _recording_tick_until_end(self, timeout=1_000):
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" ")[0]
    if os.environ.get("TICK_TRACE_FILTER", "") not in test_name:
        return _original_tick_until_end(self, timeout)
    world = self.context.world
    initial_positions = [float(v) for v in world.state.positions]
    ticks = []
    original_tick = self.tick

    def tick():
        solves_before = len(_solves)
        original_tick()
        ticks.append(
            dict(
                positions=[float(v) for v in world.state.positions],
                velocities=[float(v) for v in world.state.velocities],
                solves=_solves[solves_before:],
                nodes={
                    node.name: str(node.life_cycle_state)
                    for node in self.statechart.nodes
                }
                if hasattr(self.statechart, "nodes")
                else {},
            )
        )

    self.tick = tick
    try:
        _original_tick_until_end(self, timeout)
    finally:
        del self.tick
        file_name = (
            test_name.replace("/", "_").replace("::", "__")
            + f"_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}_{os.getpid()}.json"
        )
        (_trace_dir / file_name).write_text(
            json.dumps(
                dict(
                    machine=platform.processor() or platform.machine(),
                    piqp_build=[
                        name for name in dir(piqp) if name.startswith("piqp_python")
                    ],
                    piqp_avx512=bool(piqp.instruction_set.avx512f),
                    piqp_avx2=bool(piqp.instruction_set.avx2),
                    degree_of_freedom_names=[
                        str(dof.name) for dof in world.degrees_of_freedom
                    ],
                    initial_positions=initial_positions,
                    ticks=ticks,
                )
            )
        )
        print(f"\nTICKS USED: {len(ticks)} of {timeout}, trace in {file_name}")


QPSolverPIQP.solver_call_explicit_interface = _recording_solver_call
QPSolverPIQP.solver_call = _recording_solver_call
StatechartExecutor.tick_until_end = _recording_tick_until_end
