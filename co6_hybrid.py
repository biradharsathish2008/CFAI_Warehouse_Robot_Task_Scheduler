"""
=============================================================
CO6: Hybrid AI Architecture — Full Integration
=============================================================
Topics Covered:
  - Combining Search + CSP + Probabilistic + Utility reasoning
  - Explainable reasoning traces
  - Failure analysis
  - Heuristic bias discussion
  - Uncertainty limitations
  - Scalability analysis
=============================================================
"""

from typing import Dict, List, Optional, Tuple
from modules.co1_environment import (WarehouseState, RobotState, Task,
                                       Position, TraceLogger)
from modules.co2_search import astar, SearchResult
from modules.co3_csp import CSPScheduler
from modules.co4_utility import compute_robot_utility, resolve_narrow_aisle_conflict
from modules.co5_probabilistic import SensorModel, bayesian_obstacle_decision


# ─────────────────────────────────────────────────────────
# CO6: EXPLAINABLE REASONING TRACE
# ─────────────────────────────────────────────────────────

class ExplainableTrace:
    """
    CO6: Records every decision with a reason.
    This makes the AI system transparent and 'viva-friendly'.
    """
    def __init__(self):
        self.steps: List[Dict] = []

    def record(self, module: str, decision: str, reason: str,
                confidence: float = 1.0):
        entry = {
            "module": module,
            "decision": decision,
            "reason": reason,
            "confidence": confidence
        }
        self.steps.append(entry)
        print(f"\n  [{module}] {decision}")
        print(f"   ↳ REASON: {reason}")
        if confidence < 1.0:
            print(f"   ↳ CONFIDENCE: {confidence:.2%}")

    def show_summary(self):
        print("\n" + "="*60)
        print("  CO6 ── EXPLAINABLE REASONING TRACE SUMMARY")
        print("="*60)
        for i, step in enumerate(self.steps, 1):
            print(f"\n  Step {i}: [{step['module']}]")
            print(f"  Decision   : {step['decision']}")
            print(f"  Reason     : {step['reason']}")
            if step['confidence'] < 1.0:
                print(f"  Confidence : {step['confidence']:.2%}")


# ─────────────────────────────────────────────────────────
# CO6: HYBRID AI PIPELINE
# ─────────────────────────────────────────────────────────

class HybridAISystem:
    """
    CO6: Combines all CO modules into one integrated AI pipeline.

    Pipeline:
      1. [CO1] Formulate environment and state
      2. [CO3] Schedule tasks using CSP
      3. [CO5] Check sensor for obstacles probabilistically
      4. [CO2] Plan path using A* (rereoute if obstacle detected)
      5. [CO4] Resolve conflicts using utility
      6. [CO6] Log reasoning trace, analyze failures
    """

    def __init__(self, state: WarehouseState):
        self.state  = state
        self.trace  = ExplainableTrace()
        self.sensor = SensorModel(true_positive_rate=0.90,
                                   false_positive_rate=0.10)
        self.results: Dict[str, SearchResult] = {}
        self.assignment: Optional[Dict[str, str]] = None

    def run(self):
        """CO6: Execute the full hybrid pipeline."""
        print("\n" + "█"*60)
        print("  CO6 ── HYBRID AI SYSTEM: FULL PIPELINE RUN")
        print("█"*60)

        # ── STEP 1: CSP Task Scheduling ─────────────────────────
        print("\n  STEP 1: Scheduling Tasks (CO3 CSP)")
        scheduler = CSPScheduler(self.state.tasks, self.state.robots)
        self.assignment = scheduler.solve()

        if self.assignment:
            self.trace.record(
                "CO3-CSP",
                f"Assigned {len(self.assignment)} tasks",
                f"Backtracking with MRV+LCV+FC found valid schedule: "
                f"{self.assignment}",
                confidence=1.0
            )
        else:
            self.trace.record(
                "CO3-CSP",
                "Scheduling FAILED",
                "No assignment satisfies all constraints. "
                "Consider adding more robots or reducing tasks.",
                confidence=0.0
            )
            self._failure_analysis("CSP found no valid schedule")
            return

        # ── STEP 2: Probabilistic Obstacle Check ─────────────────
        print("\n  STEP 2: Checking Paths for Obstacles (CO5 Bayesian)")
        reroute_needed = {}
        for task_id, robot_id in self.assignment.items():
            robot = self.state.robots[robot_id]
            task  = self.state.tasks[task_id]

            # Simulate: sensor fires with 40% probability
            import random
            random.seed(42 + hash(task_id))
            sensor_fired = random.random() < 0.4
            prior = 0.25  # prior belief: 25% chance of uncharted obstacle

            posterior = self.sensor.posterior(prior, sensor_fired)
            should_reroute = posterior > 0.60

            reroute_needed[task_id] = should_reroute
            confidence = 1.0 - abs(posterior - 0.5) * 2

            self.trace.record(
                "CO5-Bayes",
                f"Path for {task_id} → {'REROUTE' if should_reroute else 'PROCEED'}",
                f"Sensor={'fired' if sensor_fired else 'silent'}, "
                f"prior={prior:.2f}, posterior={posterior:.4f}, "
                f"threshold=0.60",
                confidence=confidence
            )

        # ── STEP 3: Path Planning with A* ───────────────────────
        print("\n  STEP 3: Planning Paths (CO2 A*)")
        for task_id, robot_id in self.assignment.items():
            robot = self.state.robots[robot_id]
            task  = self.state.tasks[task_id]

            print(f"\n  Planning path for {robot_id} → {task_id}")
            result = astar(self.state, robot.position, task.pickup, verbose=True)
            self.results[task_id] = result

            if result.found:
                self.trace.record(
                    "CO2-A*",
                    f"Path found for {task_id}: {len(result.path)} steps, "
                    f"cost={result.path_cost}",
                    f"A* explored {result.nodes_expanded} nodes in "
                    f"{result.runtime_ms:.2f}ms",
                    confidence=1.0
                )
            else:
                self.trace.record(
                    "CO2-A*",
                    f"No path found for {task_id}!",
                    f"Goal unreachable from {robot.position} to {task.pickup}. "
                    f"Possible causes: obstacles surrounding goal.",
                    confidence=0.0
                )
                self._failure_analysis(f"A* failed for {task_id}")

        # ── STEP 4: Conflict Resolution ──────────────────────────
        print("\n  STEP 4: Resolving Multi-Agent Conflicts (CO4 Utility)")
        robot_list = list(self.state.robots.values())
        if len(robot_list) >= 2:
            task_list = list(self.state.tasks.values())
            aisle = Position(self.state.grid_rows // 2, self.state.grid_cols // 2)
            winner = resolve_narrow_aisle_conflict(
                robot_list[0], robot_list[1], aisle,
                task_list[0] if task_list else None,
                task_list[1] if len(task_list) > 1 else None
            )
            self.trace.record(
                "CO4-Utility",
                f"Conflict resolved: {winner} proceeds first",
                f"Higher urgency score wins narrow aisle priority",
                confidence=0.85
            )

        # ── STEP 5: Performance Metrics ──────────────────────────
        self._display_performance_metrics()

        # ── STEP 6: Reasoning Trace ──────────────────────────────
        self.trace.show_summary()

        # ── STEP 7: Critical Analysis ────────────────────────────
        self._critical_analysis()

    def _failure_analysis(self, reason: str):
        """CO6: Analyze and explain why the system failed."""
        print(f"\n  CO6 ── FAILURE ANALYSIS")
        print(f"  {'─'*50}")
        print(f"  Failure: {reason}")
        print(f"  Possible causes:")
        print(f"    1. Insufficient robots for task count")
        print(f"    2. Battery levels too low for task distances")
        print(f"    3. Obstacle configuration blocks all paths")
        print(f"  Recovery strategies:")
        print(f"    1. Add charging stations, wait for battery recharge")
        print(f"    2. Increase robot count (scalability solution)")
        print(f"    3. Recompute obstacle map (remove false-positive blocks)")

    def _display_performance_metrics(self):
        """CO6: Aggregate performance metrics across all algorithms."""
        print("\n" + "="*60)
        print("  CO6 ── SYSTEM PERFORMANCE METRICS")
        print("="*60)
        total_nodes = sum(r.nodes_expanded for r in self.results.values())
        total_time  = sum(r.runtime_ms for r in self.results.values())
        paths_found = sum(1 for r in self.results.values() if r.found)
        total_cost  = sum(r.path_cost for r in self.results.values() if r.found)

        print(f"  Tasks scheduled      : {len(self.assignment or {})}")
        print(f"  Paths found          : {paths_found} / {len(self.results)}")
        print(f"  Total nodes expanded : {total_nodes}")
        print(f"  Total planning time  : {total_time:.3f} ms")
        print(f"  Total path cost      : {total_cost} steps")
        if self.assignment:
            tasks_done = len(self.assignment)
            efficiency = (paths_found / max(tasks_done, 1)) * 100
            print(f"  System efficiency    : {efficiency:.1f}%")

    def _critical_analysis(self):
        """CO6: Academic discussion of limitations and scalability."""
        print("\n" + "─"*60)
        print("  CO6 ── CRITICAL ANALYSIS & LIMITATIONS")
        print("─"*60)
        print("""
  HEURISTIC BIAS:
  ───────────────
  • Manhattan heuristic assumes no obstacles between cells.
  • In dense obstacle environments, h(n) becomes less admissible.
  • Bias: A* may explore near-optimal paths but not absolutely optimal.
  • Fix: Use Euclidean distance or pre-computed obstacle-aware heuristics.

  UNCERTAINTY LIMITATIONS:
  ────────────────────────
  • Bayesian model assumes sensor independence across timesteps.
  • Real sensors have correlated noise (nearby cells affect each other).
  • Our prior (P=0.25) is a fixed assumption — should be learned from data.
  • HMM tracking was described but not fully implemented (scope limitation).

  SCALABILITY DISCUSSION:
  ───────────────────────
  • BFS/DFS: exponential time O(b^d) — not scalable for large warehouses.
  • A*: much better, but still O(b^d) worst case without good heuristics.
  • CSP backtracking: O(d^n) — exponential in number of tasks.
  • Scalable alternatives:
      → Hierarchical pathfinding (HPA*)
      → Genetic algorithms for scheduling (100s of tasks)
      → Deep Reinforcement Learning (warehouse robots in practice)
      → Multi-Agent systems with communication protocols (FIPA)

  REAL-WORLD GAPS:
  ────────────────
  • Robot dynamics (inertia, turning radius) not modeled.
  • Time-varying obstacles (workers walking) not handled.
  • Communication delays between robots ignored.
  • This simulation assumes perfect task completion (no physical failures).
        """)


# ─────────────────────────────────────────────────────────
# CO6: DISPLAY CO MAPPING
# ─────────────────────────────────────────────────────────

def display_co_mapping():
    """CO6: Academic CO-to-module mapping for viva reference."""
    print("\n" + "="*60)
    print("  CO6 ── COURSE OUTCOME TO MODULE MAPPING")
    print("="*60)
    mapping = [
        ("CO1", "co1_environment.py",
         "PEAS, State, Actions, Transitions, Graph, Dataclasses"),
        ("CO2", "co2_search.py",
         "BFS, DFS, UCS, A*, Heuristics, Node comparison"),
        ("CO3", "co3_csp.py",
         "Backtracking, MRV, Degree, LCV, Forward Checking"),
        ("CO4", "co4_utility.py",
         "Utility function, Multi-agent, Minimax, Alpha-Beta"),
        ("CO5", "co5_probabilistic.py",
         "Bayes Rule, Posterior, HMM, Expected Utility"),
        ("CO6", "co6_hybrid.py",
         "Full integration, Traces, Failure analysis, Scalability"),
    ]
    for co, module, topics in mapping:
        print(f"\n  {co} → {module}")
        print(f"       Topics: {topics}")
    print()
