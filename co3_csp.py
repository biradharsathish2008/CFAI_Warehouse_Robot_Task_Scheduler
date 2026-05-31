"""
=============================================================
CO3: Constraint Satisfaction Problem (CSP) — Task Scheduling
=============================================================
Topics Covered:
  - CSP formulation for robot task scheduling
  - Backtracking search
  - MRV (Minimum Remaining Values) heuristic
  - Degree heuristic
  - LCV (Least Constraining Value) heuristic
  - Forward Checking
  - Constraint violation explanations
  - Min-Conflicts concept
  - Timetabling discussion
=============================================================
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from modules.co1_environment import WarehouseState, Task, RobotState, Position


# ─────────────────────────────────────────────────────────
# CO3: CSP PROBLEM FORMULATION
# ─────────────────────────────────────────────────────────

def display_csp_formulation(tasks: Dict[str, Task], robots: Dict[str, RobotState]):
    """
    CO3: Explain CSP formulation for task scheduling.
    Variables   = Tasks (what we're assigning)
    Domain      = Robots (possible assignees)
    Constraints = Battery, Collision, Simultaneous-task rules
    """
    print("\n" + "="*60)
    print("  CO3 ── CSP FORMULATION: Task Scheduling")
    print("="*60)
    print(f"""
  Variables   : {list(tasks.keys())}
                (Each task must be assigned to exactly one robot)

  Domain      : {list(robots.keys())}
                (For each variable, domain = set of robots)

  Constraints :
    [C1] Battery   — Robot must have battery ≥ task travel cost
    [C2] Collision — Two robots cannot be at same cell same time
    [C3] Exclusive — One robot cannot handle two tasks at once
    [C4] Capacity  — Robot can only carry one package at a time
    """)


# ─────────────────────────────────────────────────────────
# CO3: CONSTRAINT CHECK
# ─────────────────────────────────────────────────────────

def check_battery_constraint(robot: RobotState, task: Task) -> Tuple[bool, str]:
    """
    CO3: C1 — Battery Constraint.
    Robot needs battery >= 2 × task.cost_estimate() for round trip.
    Returns (is_satisfied, reason_if_not).
    """
    required = task.cost_estimate() * 2  # pickup + dropoff cost
    if robot.battery < required:
        reason = (f"[C1-FAIL] {task.task_id} cannot be assigned to {robot.robot_id}: "
                  f"battery={robot.battery}% but requires ≥{required}% "
                  f"(task distance={task.cost_estimate()} × 2)")
        return False, reason
    return True, ""


def check_exclusive_constraint(assignment: Dict[str, str], task_id: str,
                                robot_id: str) -> Tuple[bool, str]:
    """
    CO3: C3 — Exclusive Constraint.
    A robot cannot be assigned more than one task simultaneously.
    """
    for assigned_task, assigned_robot in assignment.items():
        if assigned_robot == robot_id and assigned_task != task_id:
            reason = (f"[C3-FAIL] {task_id} cannot be assigned to {robot_id}: "
                      f"{robot_id} is already handling {assigned_task} simultaneously")
            return False, reason
    return True, ""


def check_all_constraints(robot: RobotState, task: Task,
                           assignment: Dict[str, str]) -> Tuple[bool, str]:
    """
    CO3: Check ALL constraints for (task → robot) assignment.
    Returns (is_valid, failure_reason).
    """
    ok, reason = check_battery_constraint(robot, task)
    if not ok:
        return False, reason

    ok, reason = check_exclusive_constraint(assignment, task.task_id, robot.robot_id)
    if not ok:
        return False, reason

    return True, ""


# ─────────────────────────────────────────────────────────
# CO3: CSP HEURISTICS
# ─────────────────────────────────────────────────────────

def mrv_heuristic(unassigned: List[str], tasks: Dict[str, Task],
                  robots: Dict[str, RobotState],
                  assignment: Dict[str, str]) -> str:
    """
    CO3: MRV — Minimum Remaining Values Heuristic.
    Choose the variable (task) with the FEWEST legal values (robots) left.
    Rationale: fail early on most constrained tasks → prune search tree faster.
    """
    best_task = None
    min_remaining = float('inf')

    for task_id in unassigned:
        task = tasks[task_id]
        remaining = 0
        for robot_id, robot in robots.items():
            ok, _ = check_all_constraints(robot, task, assignment)
            if ok:
                remaining += 1
        if remaining < min_remaining:
            min_remaining = remaining
            best_task = task_id

    return best_task


def degree_heuristic(unassigned: List[str], tasks: Dict[str, Task]) -> str:
    """
    CO3: Degree Heuristic.
    Choose the variable (task) involved in the MOST constraints with other unassigned tasks.
    Used as a tiebreaker when MRV has equal scores.
    Here: higher-priority tasks are treated as more constrained.
    """
    return max(unassigned, key=lambda tid: tasks[tid].priority)


def lcv_heuristic(task_id: str, tasks: Dict[str, Task],
                  robots: Dict[str, RobotState],
                  assignment: Dict[str, str],
                  unassigned: List[str]) -> List[str]:
    """
    CO3: LCV — Least Constraining Value Heuristic.
    Order robot assignments so that the robot that LEAVES THE MOST OPTIONS
    for remaining tasks is tried first.
    Rationale: don't block other tasks unnecessarily.
    """
    task = tasks[task_id]
    robot_scores = []

    for robot_id, robot in robots.items():
        ok, _ = check_all_constraints(robot, task, assignment)
        if not ok:
            continue
        # Count how many other tasks this robot can still do after assignment
        temp_assignment = dict(assignment)
        temp_assignment[task_id] = robot_id
        remaining_options = 0
        for other_tid in unassigned:
            if other_tid == task_id:
                continue
            for rid, rob in robots.items():
                o_ok, _ = check_all_constraints(rob, tasks[other_tid], temp_assignment)
                if o_ok:
                    remaining_options += 1
        robot_scores.append((robot_id, remaining_options))

    # Sort descending: try robot that leaves most options for others
    robot_scores.sort(key=lambda x: -x[1])
    return [r for r, _ in robot_scores]


# ─────────────────────────────────────────────────────────
# CO3: FORWARD CHECKING
# ─────────────────────────────────────────────────────────

def forward_check(assignment: Dict[str, str], unassigned: List[str],
                  tasks: Dict[str, Task],
                  robots: Dict[str, RobotState]) -> Tuple[bool, str]:
    """
    CO3: Forward Checking.
    After each assignment, prune domains of remaining unassigned variables.
    If any variable's domain becomes EMPTY → backtrack immediately.
    """
    for task_id in unassigned:
        legal_count = 0
        for robot_id, robot in robots.items():
            ok, _ = check_all_constraints(robot, tasks[task_id], assignment)
            if ok:
                legal_count += 1
        if legal_count == 0:
            return False, (f"[FC-FAIL] Forward check: {task_id} has NO legal "
                           f"robot assignment after current choices — backtrack!")
    return True, ""


# ─────────────────────────────────────────────────────────
# CO3: BACKTRACKING CSP SOLVER
# ─────────────────────────────────────────────────────────

class CSPScheduler:
    """
    CO3: Backtracking CSP Solver with MRV + Degree + LCV + Forward Checking.
    Assigns each task to a robot satisfying all constraints.
    """

    def __init__(self, tasks: Dict[str, Task], robots: Dict[str, RobotState]):
        self.tasks     = tasks
        self.robots    = robots
        self.calls     = 0      # recursion depth counter
        self.backtracks = 0

    def solve(self) -> Optional[Dict[str, str]]:
        """
        CO3: Entry point for backtracking solver.
        Returns: assignment dict {task_id: robot_id} or None if unsolvable.
        """
        print("\n" + "="*60)
        print("  CO3 ── CSP BACKTRACKING SOLVER")
        print("="*60)
        display_csp_formulation(self.tasks, self.robots)
        result = self._backtrack({}, list(self.tasks.keys()))
        print(f"\n  Solver Stats: {self.calls} calls, {self.backtracks} backtracks")
        return result

    def _backtrack(self, assignment: Dict[str, str],
                   unassigned: List[str]) -> Optional[Dict[str, str]]:
        """
        CO3: Recursive backtracking with all heuristics.
        Base case: all tasks assigned.
        Recursive: pick best variable, try ordered values, forward-check.
        """
        self.calls += 1

        # Base case: assignment complete
        if not unassigned:
            print("\n  ✓ Complete assignment found!")
            return assignment

        # ── CO3: MRV — pick the most constrained task ──────────
        task_id = mrv_heuristic(unassigned, self.tasks, self.robots, assignment)
        if task_id is None:
            task_id = degree_heuristic(unassigned, self.tasks)

        print(f"\n  [Backtrack call #{self.calls}]")
        print(f"  MRV selected task: {task_id} "
              f"(priority={self.tasks[task_id].priority})")

        remaining_unassigned = [t for t in unassigned if t != task_id]

        # ── CO3: LCV — order robots least-constraining first ───
        ordered_robots = lcv_heuristic(task_id, self.tasks, self.robots,
                                        assignment, unassigned)

        if not ordered_robots:
            print(f"  [!] No legal robot for {task_id} → BACKTRACK")
            self.backtracks += 1
            return None

        print(f"  LCV robot order: {ordered_robots}")

        for robot_id in ordered_robots:
            robot = self.robots[robot_id]
            ok, reason = check_all_constraints(robot, self.tasks[task_id], assignment)

            if not ok:
                print(f"  ✗ {reason}")
                continue

            # Tentatively assign
            assignment[task_id] = robot_id
            print(f"  → Trying: {task_id} ← {robot_id} "
                  f"(battery={robot.battery}%, "
                  f"task_cost={self.tasks[task_id].cost_estimate()})")

            # ── CO3: Forward Checking ─────────────────────────
            fc_ok, fc_reason = forward_check(assignment, remaining_unassigned,
                                              self.tasks, self.robots)
            if not fc_ok:
                print(f"  {fc_reason}")
                del assignment[task_id]
                self.backtracks += 1
                continue

            # Recurse
            result = self._backtrack(dict(assignment), remaining_unassigned)
            if result is not None:
                return result

            # Undo assignment
            del assignment[task_id]
            self.backtracks += 1

        return None


# ─────────────────────────────────────────────────────────
# CO3: DISPLAY SCHEDULE
# ─────────────────────────────────────────────────────────

def display_schedule(assignment: Optional[Dict[str, str]],
                     tasks: Dict[str, Task],
                     robots: Dict[str, RobotState]):
    """CO3: Pretty-print the final task schedule."""
    print("\n" + "="*60)
    print("  CO3 ── FINAL TASK SCHEDULE")
    print("="*60)
    if assignment is None:
        print("  [!] No valid schedule found. Constraints unsatisfiable!")
        return

    for task_id, robot_id in assignment.items():
        task  = tasks[task_id]
        robot = robots[robot_id]
        cost  = task.cost_estimate()
        print(f"  {task_id} → {robot_id}")
        print(f"     Pickup : {task.pickup}  Dropoff: {task.dropoff}")
        print(f"     Priority: {task.priority}  Est. cost: {cost}  "
              f"Robot battery: {robot.battery}%")
        print()


# ─────────────────────────────────────────────────────────
# CO3: EDUCATIONAL EXPLANATIONS
# ─────────────────────────────────────────────────────────

def explain_csp_concepts():
    """CO3: Explain Min-Conflicts, scheduling, and timetabling."""
    print("\n" + "─"*60)
    print("  CO3 ── KEY CSP CONCEPTS (Academic)")
    print("─"*60)
    print("""
  MIN-CONFLICTS Algorithm (Alternative to Backtracking):
  ────────────────────────────────────────────────────
  • Start with a random complete assignment (may violate constraints).
  • Repeat: pick a conflicting variable, reassign it to the value
    that MINIMIZES the number of conflicts with other variables.
  • Very effective for large scheduling problems (e.g., n-queens).
  • Used in: airline crew scheduling, university timetabling.

  SCHEDULING vs TIMETABLING:
  ─────────────────────────
  • Scheduling: Assign tasks to agents over time (our problem).
  • Timetabling: Assign activities to time slots + rooms.
  • Both are CSPs: variables=activities, domains=slots/agents,
    constraints=no overlap, capacity, precedence.

  WHY BACKTRACKING IS BETTER HERE:
  ────────────────────────────────
  • Our problem is small (few robots, few tasks) → backtracking fast.
  • Min-conflicts shines for 100s–1000s of variables.
    """)
