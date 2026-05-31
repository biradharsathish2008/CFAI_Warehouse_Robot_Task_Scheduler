"""
=============================================================
CO1: Problem Formulation & Environment Representation
=============================================================
Topics Covered:
  - PEAS Model (Performance, Environment, Actuators, Sensors)
  - Environment Types (Grid-based, Partially Observable, Multi-Agent)
  - State Representation using Python dataclasses
  - Action Representation
  - Transition Model
  - Cost Function
  - Graph Representation
  - Trace Logging for academic clarity
=============================================================
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import time

# ─────────────────────────────────────────────────────────
# CO1: PEAS MODEL
# ─────────────────────────────────────────────────────────

def display_peas_model():
    """
    CO1: Display the PEAS description of the Warehouse Robot Agent.
    PEAS = Performance | Environment | Actuators | Sensors
    """
    print("\n" + "="*60)
    print("  CO1 ── PEAS MODEL: Warehouse Robot Agent")
    print("="*60)
    print("""
  ┌──────────────────────────────────────────────────────┐
  │  P ─ Performance Measure                            │
  │      • Packages delivered per unit time             │
  │      • Battery consumed per trip                    │
  │      • Collisions avoided                           │
  │      • Path length (fewer steps = better)           │
  ├──────────────────────────────────────────────────────┤
  │  E ─ Environment                                    │
  │      • Grid-based warehouse (rows × cols)           │
  │      • Static obstacles (shelves, walls)            │
  │      • Dynamic: other robots, uncertain sensors     │
  │      • Partially observable (sensor noise)          │
  ├──────────────────────────────────────────────────────┤
  │  A ─ Actuators                                      │
  │      • Move: UP / DOWN / LEFT / RIGHT               │
  │      • Pick Package                                 │
  │      • Drop Package                                 │
  │      • Wait (charge / yield)                        │
  ├──────────────────────────────────────────────────────┤
  │  S ─ Sensors                                        │
  │      • GPS-like position sensor                     │
  │      • Proximity sensor (detects obstacles)         │
  │      • Battery level monitor                        │
  │      • Package detection sensor                     │
  └──────────────────────────────────────────────────────┘
    """)

# ─────────────────────────────────────────────────────────
# CO1: ENVIRONMENT TYPES
# ─────────────────────────────────────────────────────────

def display_environment_types():
    """
    CO1: Classify the warehouse environment using standard AI taxonomy.
    """
    print("\n" + "─"*60)
    print("  CO1 ── ENVIRONMENT TYPES")
    print("─"*60)
    types = [
        ("Observable",     "Partially Observable", "Robots sense only nearby cells"),
        ("Deterministic",  "Stochastic",           "Sensor noise adds uncertainty"),
        ("Episodic",       "Sequential",           "Each move affects future states"),
        ("Static",         "Semi-Dynamic",         "Obstacles fixed; robots move"),
        ("Discrete",       "Discrete",             "Grid cells, discrete actions"),
        ("Single-Agent",   "Multi-Agent",          "Multiple robots cooperate"),
    ]
    print(f"  {'Property':<18} {'Classification':<20} {'Reason'}")
    print(f"  {'─'*17} {'─'*19} {'─'*25}")
    for prop, cls, reason in types:
        print(f"  {prop:<18} {cls:<20} {reason}")
    print()

# ─────────────────────────────────────────────────────────
# CO1: STATE REPRESENTATION (Python Dataclass)
# ─────────────────────────────────────────────────────────

@dataclass
class Position:
    """CO1: Represents a 2D grid coordinate (row, col)."""
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return f"({self.row},{self.col})"

    def to_tuple(self) -> Tuple[int, int]:
        return (self.row, self.col)


@dataclass
class Task:
    """
    CO1: A delivery task.
    - task_id: Unique identifier (T1, T2, ...)
    - pickup: Grid cell where package is picked up
    - dropoff: Grid cell where package is delivered
    - priority: 1=low, 2=medium, 3=high
    - assigned_to: Robot ID (None if unassigned)
    """
    task_id: str
    pickup: Position
    dropoff: Position
    priority: int = 1
    assigned_to: Optional[str] = None

    def cost_estimate(self) -> int:
        """CO1: Manhattan distance as cost estimate for this task."""
        return (abs(self.pickup.row - self.dropoff.row) +
                abs(self.pickup.col - self.dropoff.col))


@dataclass
class RobotState:
    """
    CO1: Complete state of a single robot.
    State = (position, battery, carrying_task, status)
    This is the state representation used by all search algorithms.
    """
    robot_id: str
    position: Position
    battery: int                        # 0–100 units
    carrying_task: Optional[str] = None # task_id if carrying a package
    status: str = "idle"                # idle | moving | charging | done

    def is_battery_ok(self, required: int = 10) -> bool:
        """CO1: Check if battery is sufficient to act."""
        return self.battery >= required

    def __repr__(self):
        return (f"Robot({self.robot_id}, pos={self.position}, "
                f"batt={self.battery}%, status={self.status})")


@dataclass
class WarehouseState:
    """
    CO1: Global state of the entire warehouse at a point in time.
    This is the 'world state' passed between modules.
    """
    robots: Dict[str, RobotState]
    tasks: Dict[str, Task]
    obstacles: Set[Tuple[int, int]]
    delivery_zone: Position
    grid_rows: int
    grid_cols: int
    timestep: int = 0

    def is_obstacle(self, pos: Position) -> bool:
        return (pos.row, pos.col) in self.obstacles

    def is_valid(self, pos: Position) -> bool:
        return (0 <= pos.row < self.grid_rows and
                0 <= pos.col < self.grid_cols and
                not self.is_obstacle(pos))


# ─────────────────────────────────────────────────────────
# CO1: ACTION REPRESENTATION
# ─────────────────────────────────────────────────────────

ACTIONS = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
    "WAIT":  ( 0,  0),
    "PICK":  ( 0,  0),
    "DROP":  ( 0,  0),
}

def apply_action(pos: Position, action: str) -> Position:
    """
    CO1: Transition model — returns new position after action.
    T(state, action) → state'
    """
    dr, dc = ACTIONS[action]
    return Position(pos.row + dr, pos.col + dc)


def action_cost(action: str, battery: int) -> int:
    """
    CO1: Cost function — every move costs 1 battery unit.
    WAIT costs 0 (robot is stationary).
    """
    if action == "WAIT":
        return 0
    return 1  # uniform step cost


# ─────────────────────────────────────────────────────────
# CO1: GRAPH REPRESENTATION
# ─────────────────────────────────────────────────────────

def build_adjacency_graph(state: WarehouseState) -> Dict[Tuple, List[Tuple]]:
    """
    CO1: Build an adjacency graph of the warehouse grid.
    Nodes = valid grid cells (not obstacles).
    Edges = valid moves between adjacent cells.
    Returns: dict mapping (row,col) → list of (row,col) neighbors
    """
    graph: Dict[Tuple, List[Tuple]] = {}
    for r in range(state.grid_rows):
        for c in range(state.grid_cols):
            pos = Position(r, c)
            if state.is_valid(pos):
                neighbors = []
                for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    npos = apply_action(pos, action)
                    if state.is_valid(npos):
                        neighbors.append(npos.to_tuple())
                graph[(r, c)] = neighbors
    return graph


def display_graph_info(graph: Dict, state: WarehouseState):
    """CO1: Display graph statistics for academic understanding."""
    print("\n  CO1 ── GRAPH REPRESENTATION")
    print(f"  ├─ Total nodes (valid cells) : {len(graph)}")
    total_edges = sum(len(v) for v in graph.values())
    print(f"  ├─ Total edges               : {total_edges}")
    blocked = state.grid_rows * state.grid_cols - len(graph)
    print(f"  ├─ Blocked cells (obstacles) : {blocked}")
    print(f"  └─ Grid dimensions           : {state.grid_rows} × {state.grid_cols}")


# ─────────────────────────────────────────────────────────
# CO1: TRACE LOGGER
# ─────────────────────────────────────────────────────────

class TraceLogger:
    """
    CO1: Step-by-step trace logger.
    Records every significant event for academic review.
    """
    def __init__(self):
        self.logs: List[str] = []
        self.start_time = time.time()

    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        entry = f"  [{elapsed:6.3f}s] {msg}"
        self.logs.append(entry)
        print(entry)

    def section(self, title: str):
        print(f"\n  {'─'*50}")
        print(f"  TRACE: {title}")
        print(f"  {'─'*50}")

    def show_state(self, robot: RobotState):
        self.log(f"State → {robot}")

    def show_transition(self, from_pos: Position, action: str, to_pos: Position, cost: int):
        self.log(f"Transition: {from_pos} ──[{action}, cost={cost}]──→ {to_pos}")
