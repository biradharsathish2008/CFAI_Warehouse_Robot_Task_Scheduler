"""
=============================================================
WAREHOUSE GRID: Visual Display & User Customization
=============================================================
Features:
  - ASCII art warehouse grid display
  - Legend for all symbols
  - Interactive user customization (robots, tasks, obstacles)
  - Grid re-rendering after changes
=============================================================
"""

from typing import Set, Dict, Tuple, List, Optional
from modules.co1_environment import (WarehouseState, RobotState, Task,
                                       Position)


# ─────────────────────────────────────────────────────────
# GRID DISPLAY
# ─────────────────────────────────────────────────────────

def display_grid(state: WarehouseState, title: str = "WAREHOUSE GRID",
                 highlight_path: Optional[List[Position]] = None):
    """
    Display the warehouse as an ASCII grid.
    Symbols:
      R1/R2 = Robot positions
      T1/T2 = Task pickup locations
      D     = Delivery zone
      X     = Obstacle/blocked cell
      *     = Path cell (if provided)
      .     = Empty cell
    """
    path_set: Set[Tuple] = set()
    if highlight_path:
        path_set = {p.to_tuple() for p in highlight_path}

    # Build lookup maps
    robot_at: Dict[Tuple, str] = {
        r.position.to_tuple(): r.robot_id for r in state.robots.values()
    }
    task_at: Dict[Tuple, str] = {
        t.pickup.to_tuple(): t.task_id for t in state.tasks.values()
    }
    dropoff_at: Dict[Tuple, str] = {
        t.dropoff.to_tuple(): t.task_id + "D" for t in state.tasks.values()
    }
    delivery = state.delivery_zone.to_tuple()

    cols = state.grid_cols
    rows = state.grid_rows

    # Cell width = 4 chars
    cell_w = 4
    border_top    = "  +" + ("─" * (cell_w * cols + cols - 1)) + "+"
    border_bottom = border_top

    print(f"\n  ╔{'═'*(len(border_top)-3)}╗")
    print(f"  ║  {title:^{len(border_top)-6}}  ║")
    print(f"  ╚{'═'*(len(border_top)-3)}╝")
    print(border_top)

    for r in range(rows):
        row_parts = []
        for c in range(cols):
            pos = (r, c)
            if pos in state.obstacles:
                cell = " X  "
            elif pos in robot_at:
                rid = robot_at[pos]
                cell = f" {rid:<3}"
            elif pos in task_at:
                tid = task_at[pos]
                cell = f" {tid:<3}"
            elif pos == delivery:
                cell = " D  "
            elif pos in path_set:
                cell = " *  "
            else:
                cell = " .  "
            row_parts.append(cell)

        # Row with row index
        row_str = "  |" + "|".join(row_parts) + f"|  row {r}"
        print(row_str)

    print(border_bottom)

    # Column indices
    col_indices = "     " + "   ".join(str(c) for c in range(cols))
    print(col_indices)

    # Legend
    print(f"\n  LEGEND:")
    print(f"    R1/R2 = Robot positions    |  T1/T2 = Task pickup")
    print(f"    D     = Delivery zone      |  X     = Obstacle")
    print(f"    *     = Path               |  .     = Empty cell")

    # Robot info
    print(f"\n  ROBOT STATUS:")
    for rid, robot in state.robots.items():
        status_bar = ("█" * (robot.battery // 10)).ljust(10, "░")
        print(f"    {rid}: pos={robot.position}  battery=[{status_bar}] "
              f"{robot.battery}%  status={robot.status}")

    # Task info
    print(f"\n  TASK LIST:")
    for tid, task in state.tasks.items():
        assigned = task.assigned_to or "Unassigned"
        print(f"    {tid}: pickup={task.pickup} → dropoff={task.dropoff}  "
              f"priority={task.priority}  assigned={assigned}")
    print()


# ─────────────────────────────────────────────────────────
# USER CUSTOMIZATION
# ─────────────────────────────────────────────────────────

def get_int_input(prompt: str, min_val: int, max_val: int,
                   default: Optional[int] = None) -> int:
    """Helper: Get validated integer input from user."""
    while True:
        try:
            if default is not None:
                val = input(f"  {prompt} (default={default}): ").strip()
                if val == "":
                    return default
            else:
                val = input(f"  {prompt}: ").strip()
            n = int(val)
            if min_val <= n <= max_val:
                return n
            print(f"  [!] Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  [!] Invalid input. Enter a number.")


def get_position_input(prompt: str, rows: int, cols: int) -> Position:
    """Helper: Get a valid grid position from user."""
    print(f"  {prompt}")
    r = get_int_input(f"    Row (0–{rows-1})", 0, rows - 1)
    c = get_int_input(f"    Col (0–{cols-1})", 0, cols - 1)
    return Position(r, c)


def customize_grid(state: WarehouseState) -> WarehouseState:
    """
    Interactive customization of warehouse:
      1. Modify robot positions
      2. Modify task positions
      3. Add/remove obstacles
    Returns the updated WarehouseState.
    """
    print("\n" + "="*60)
    print("  WAREHOUSE CUSTOMIZATION")
    print("="*60)
    print("  You can customize the warehouse for different scenarios.")
    rows, cols = state.grid_rows, state.grid_cols

    # ── 1. Robot positions ────────────────────────────────
    print(f"\n  [1] ROBOT POSITIONS")
    modify_robots = input("  Do you want to modify robot positions? (y/n): ").strip().lower()
    if modify_robots == "y":
        for rid in list(state.robots.keys()):
            print(f"\n  Current position of {rid}: {state.robots[rid].position}")
            new_pos = get_position_input(
                f"Enter new position for {rid}:", rows, cols
            )
            # Check for obstacle
            if new_pos.to_tuple() in state.obstacles:
                print(f"  [!] That cell is an obstacle. Keeping original position.")
            else:
                state.robots[rid].position = new_pos
                print(f"  ✓ {rid} moved to {new_pos}")

            # Battery
            new_batt = get_int_input(
                f"Battery for {rid}% (0–100)", 0, 100,
                default=state.robots[rid].battery
            )
            state.robots[rid].battery = new_batt

    # ── 2. Task positions ─────────────────────────────────
    print(f"\n  [2] TASK POSITIONS")
    modify_tasks = input("  Do you want to modify task positions? (y/n): ").strip().lower()
    if modify_tasks == "y":
        for tid in list(state.tasks.keys()):
            task = state.tasks[tid]
            print(f"\n  Task {tid}: pickup={task.pickup} → dropoff={task.dropoff}")
            change = input(f"  Modify {tid}? (y/n): ").strip().lower()
            if change == "y":
                print(f"  New pickup position for {tid}:")
                new_pickup = get_position_input("", rows, cols)
                print(f"  New dropoff position for {tid}:")
                new_dropoff = get_position_input("", rows, cols)
                new_priority = get_int_input(
                    f"Priority for {tid} (1=low, 2=med, 3=high)", 1, 3,
                    default=task.priority
                )
                state.tasks[tid].pickup   = new_pickup
                state.tasks[tid].dropoff  = new_dropoff
                state.tasks[tid].priority = new_priority
                print(f"  ✓ {tid} updated")

    # ── 3. Obstacles ──────────────────────────────────────
    print(f"\n  [3] OBSTACLES")
    print(f"  Current obstacles: {sorted(state.obstacles)}")
    modify_obs = input("  Do you want to add/remove obstacles? (y/n): ").strip().lower()
    if modify_obs == "y":
        while True:
            print("\n    a) Add obstacle")
            print("    r) Remove obstacle")
            print("    d) Done")
            choice = input("    Choice: ").strip().lower()
            if choice == "a":
                pos = get_position_input("Enter obstacle position:", rows, cols)
                pt  = pos.to_tuple()
                # Don't block robots or tasks
                blocked = set()
                for rob in state.robots.values():
                    blocked.add(rob.position.to_tuple())
                for t in state.tasks.values():
                    blocked.add(t.pickup.to_tuple())
                    blocked.add(t.dropoff.to_tuple())
                blocked.add(state.delivery_zone.to_tuple())
                if pt in blocked:
                    print("    [!] Cannot place obstacle on robot/task/delivery.")
                else:
                    state.obstacles.add(pt)
                    print(f"    ✓ Obstacle added at {pos}")
            elif choice == "r":
                pos = get_position_input("Enter obstacle to remove:", rows, cols)
                pt  = pos.to_tuple()
                if pt in state.obstacles:
                    state.obstacles.remove(pt)
                    print(f"    ✓ Obstacle removed at {pos}")
                else:
                    print("    [!] No obstacle at that position.")
            elif choice == "d":
                break
            else:
                print("    [!] Invalid choice.")

    print("\n  ✓ Customization complete. Rendering updated grid...")
    return state
