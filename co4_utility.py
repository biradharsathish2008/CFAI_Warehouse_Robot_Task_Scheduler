"""
=============================================================
CO4: Utility-Based Agents & Multi-Agent Decision Making
=============================================================
Topics Covered:
  - Utility function design
  - Multi-agent conflict resolution (narrow aisle scenario)
  - Minimax intuition
  - Alpha-Beta Pruning explanation
  - Bounded Rationality discussion
=============================================================
"""

from typing import Dict, List, Tuple, Optional
from modules.co1_environment import Position, RobotState, Task, WarehouseState


# ─────────────────────────────────────────────────────────
# CO4: UTILITY FUNCTION
# ─────────────────────────────────────────────────────────

def compute_robot_utility(robot: RobotState, task: Task,
                           state: WarehouseState) -> float:
    """
    CO4: Utility function U(robot, task).
    Higher utility = better candidate to handle the task.

    Factors considered:
      + Battery level (more = better, can complete task)
      + Proximity to pickup (closer = better)
      - Task cost (shorter task = better use of resources)
      + Priority of task (higher = more urgent)

    Returns a float utility score.
    """
    battery_score  = robot.battery / 100.0             # 0.0–1.0
    distance       = (abs(robot.position.row - task.pickup.row) +
                      abs(robot.position.col - task.pickup.col))
    proximity_score = 1.0 / (1.0 + distance)           # 0.0–1.0
    task_cost_score = 1.0 / (1.0 + task.cost_estimate()) # prefer short tasks
    priority_score  = task.priority / 3.0              # 0.0–1.0

    # Weighted sum: battery is critical, proximity matters most operationally
    utility = (0.30 * battery_score +
               0.40 * proximity_score +
               0.10 * task_cost_score +
               0.20 * priority_score)

    return round(utility, 4)


def display_utility_matrix(robots: Dict[str, RobotState],
                            tasks: Dict[str, Task],
                            state: WarehouseState):
    """
    CO4: Display utility matrix — rows=robots, cols=tasks.
    Shows which robot is best suited for each task.
    """
    print("\n" + "="*60)
    print("  CO4 ── UTILITY MATRIX (robot × task)")
    print("="*60)
    task_ids = list(tasks.keys())

    # Header row
    header = f"  {'Robot':<10}"
    for tid in task_ids:
        header += f" {tid:>8}"
    print(header)
    print(f"  {'─'*9}" + "─" * (9 * len(task_ids)))

    for robot_id, robot in robots.items():
        row = f"  {robot_id:<10}"
        for tid in task_ids:
            u = compute_robot_utility(robot, tasks[tid], state)
            row += f" {u:>8.4f}"
        print(row)

    print(f"\n  Best assignments by utility:")
    for tid, task in tasks.items():
        best_robot = max(robots.keys(),
                         key=lambda rid: compute_robot_utility(robots[rid], task, state))
        best_u = compute_robot_utility(robots[best_robot], task, state)
        print(f"  {tid} → {best_robot} (utility={best_u:.4f})")


# ─────────────────────────────────────────────────────────
# CO4: MULTI-AGENT CONFLICT RESOLUTION
# ─────────────────────────────────────────────────────────

def resolve_narrow_aisle_conflict(robot1: RobotState, robot2: RobotState,
                                   aisle_pos: Position,
                                   task1: Optional[Task] = None,
                                   task2: Optional[Task] = None) -> str:
    """
    CO4: Multi-Agent Decision — Two robots approaching the same narrow aisle.
    The system evaluates utilities and decides which robot proceeds first.

    Scenario:
      - Robot1 and Robot2 both need to pass through aisle_pos.
      - Only one can pass at a time (capacity=1).
      - We compare their urgency scores to decide priority.

    Returns: robot_id of robot that proceeds first.
    """
    print("\n" + "="*60)
    print("  CO4 ── MULTI-AGENT CONFLICT: Narrow Aisle")
    print("="*60)
    print(f"  Aisle position  : {aisle_pos}")
    print(f"  Competing robots: {robot1.robot_id}, {robot2.robot_id}")
    print()

    def urgency(robot: RobotState, task: Optional[Task]) -> float:
        """
        CO4: Urgency = combination of task priority, battery anxiety, and proximity.
        Robots with high-priority tasks or low battery get higher urgency.
        """
        priority_factor = (task.priority / 3.0) if task else 0.5
        # Low battery → more urgent (can't afford to wait and waste energy)
        battery_anxiety = 1.0 - (robot.battery / 100.0)
        dist_to_aisle = (abs(robot.position.row - aisle_pos.row) +
                         abs(robot.position.col - aisle_pos.col))
        proximity = 1.0 / (1.0 + dist_to_aisle)
        return 0.4 * priority_factor + 0.3 * battery_anxiety + 0.3 * proximity

    u1 = round(urgency(robot1, task1), 4)
    u2 = round(urgency(robot2, task2), 4)

    print(f"  Urgency scores:")
    print(f"    {robot1.robot_id}: {u1}  (battery={robot1.battery}%, "
          f"task_priority={task1.priority if task1 else 'None'})")
    print(f"    {robot2.robot_id}: {u2}  (battery={robot2.battery}%, "
          f"task_priority={task2.priority if task2 else 'None'})")
    print()

    if u1 >= u2:
        winner = robot1.robot_id
        loser  = robot2.robot_id
    else:
        winner = robot2.robot_id
        loser  = robot1.robot_id

    print(f"  DECISION: {winner} proceeds through aisle first.")
    print(f"  REASON  : Higher urgency score ({max(u1,u2):.4f} vs {min(u1,u2):.4f})")
    print(f"  ACTION  : {loser} waits 1 timestep (WAIT action).")
    return winner


# ─────────────────────────────────────────────────────────
# CO4: MINIMAX INTUITION
# ─────────────────────────────────────────────────────────

def explain_minimax():
    """
    CO4: Minimax algorithm intuition with example.
    In a competitive warehouse scenario, robots might compete for the
    same charging station. We model this as a 2-player zero-sum game.
    """
    print("\n" + "─"*60)
    print("  CO4 ── MINIMAX INTUITION")
    print("─"*60)
    print("""
  Minimax models adversarial decision making:
  ─────────────────────────────────────────
  • MAX player (Robot A) wants to MAXIMIZE its utility.
  • MIN player (Robot B) wants to MINIMIZE Robot A's utility
    (because both competing for the same resource).

  Example: Two robots competing for one charging station:

              [Root: Robot A decides]
              /                     \\
         Go LEFT                  Go RIGHT
        (value=3)                 (value=5)
         /     \\                  /       \\
    Wait  Charge            Wait    Charge
    (1)    (3)              (4)      (5)
              ↑ MIN picks 3           ↑ MIN picks 4
         ↑ MAX picks RIGHT branch (value=4)

  Alpha-Beta Pruning:
  ──────────────────
  α = best value MAX can guarantee so far (lower bound)
  β = best value MIN can guarantee so far (upper bound)

  If at any point α ≥ β → prune (no need to explore further).
  This reduces O(b^d) to O(b^(d/2)) in the best case.
  Example: 1,000,000 nodes → 1,000 nodes with pruning!

  Bounded Rationality:
  ───────────────────
  • Real robots have limited time and computation.
  • Instead of exploring full minimax tree (depth d),
    apply a depth limit L and use an EVALUATION FUNCTION
    at leaf nodes (like our utility function).
  • This is called: H-Minimax (Heuristic Minimax).
  • Used in chess engines (e.g., Stockfish limits to depth 20).
    """)


# ─────────────────────────────────────────────────────────
# CO4: SIMPLIFIED MINIMAX (2-level, demonstrative)
# ─────────────────────────────────────────────────────────

def minimax_demo(robot: RobotState, options: List[Dict]) -> Dict:
    """
    CO4: Simplified 2-level minimax for robot decision demo.
    options = [{"action": "...", "max_val": ..., "min_response": ...}, ...]
    Robot (MAX) picks action, adversary (MIN) picks worst outcome.
    Robot selects the action that maximizes the minimum outcome.
    """
    print("\n  CO4 ── MINIMAX DEMO (2-level)")
    print(f"  Robot {robot.robot_id} evaluating {len(options)} actions:")

    best_action = None
    best_value  = float('-inf')

    for opt in options:
        min_val = opt["min_response"]  # adversary minimizes
        print(f"    Action={opt['action']:10s} | "
              f"MAX_val={opt['max_val']} | "
              f"MIN_response={min_val} | minimax_val={min_val}")
        if min_val > best_value:
            best_value  = min_val
            best_action = opt["action"]

    print(f"  → Best action: {best_action} (minimax value={best_value})")
    return {"action": best_action, "value": best_value}


# ─────────────────────────────────────────────────────────
# CO4: RUN CO4 DEMO
# ─────────────────────────────────────────────────────────

def run_co4_demo(state: WarehouseState):
    """CO4: Run all CO4 demonstrations."""
    robots = state.robots
    tasks  = state.tasks

    # 1. Utility matrix
    display_utility_matrix(robots, tasks, state)

    # 2. Narrow aisle conflict
    robot_list = list(robots.values())
    task_list  = list(tasks.values())
    if len(robot_list) >= 2:
        aisle = Position(state.grid_rows // 2, state.grid_cols // 2)
        resolve_narrow_aisle_conflict(
            robot_list[0], robot_list[1], aisle,
            task_list[0] if task_list else None,
            task_list[1] if len(task_list) > 1 else None
        )

    # 3. Minimax demo
    r1 = robot_list[0]
    options = [
        {"action": "GO_LEFT",   "max_val": 7, "min_response": 3},
        {"action": "GO_RIGHT",  "max_val": 9, "min_response": 5},
        {"action": "WAIT",      "max_val": 4, "min_response": 4},
    ]
    minimax_demo(r1, options)

    # 4. Educational explanation
    explain_minimax()
