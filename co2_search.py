"""
=============================================================
CO2: Uninformed & Informed Search Algorithms
=============================================================
Topics Covered:
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Uniform Cost Search (UCS)
  - A* Search with Manhattan heuristic
  - Open/Closed set visualization
  - Node expansions & runtime comparison
  - Optimal path display on grid
=============================================================
"""

import time
import heapq
import sys
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from modules.co1_environment import Position, WarehouseState, apply_action, ACTIONS


# ─────────────────────────────────────────────────────────
# CO2: HEURISTIC FUNCTION
# ─────────────────────────────────────────────────────────

def manhattan_heuristic(pos: Position, goal: Position) -> int:
    """
    CO2: Manhattan Distance heuristic h(n).
    Admissible: never overestimates (grid allows only 4-directional moves).
    h(n) = |row_n - row_goal| + |col_n - col_goal|
    """
    return abs(pos.row - goal.row) + abs(pos.col - goal.col)


# ─────────────────────────────────────────────────────────
# CO2: SEARCH RESULT DATACLASS
# ─────────────────────────────────────────────────────────

class SearchResult:
    """CO2: Stores results of any search algorithm for comparison."""
    def __init__(self, algorithm: str):
        self.algorithm      = algorithm
        self.path           : List[Position] = []
        self.explored_nodes : List[Position] = []
        self.open_set_sizes : List[int] = []
        self.nodes_expanded : int = 0
        self.path_cost      : int = 0
        self.runtime_ms     : float = 0.0
        self.found          : bool = False
        self.memory_est_kb  : float = 0.0

    def summary(self):
        print(f"\n  {'─'*55}")
        print(f"  Algorithm     : {self.algorithm}")
        print(f"  Path Found    : {'YES ✓' if self.found else 'NO ✗'}")
        print(f"  Path Length   : {len(self.path)} steps")
        print(f"  Path Cost     : {self.path_cost}")
        print(f"  Nodes Expanded: {self.nodes_expanded}")
        print(f"  Runtime       : {self.runtime_ms:.3f} ms")
        print(f"  Memory Est.   : {self.memory_est_kb:.2f} KB")
        if self.path:
            path_str = " → ".join(str(p) for p in self.path[:6])
            if len(self.path) > 6:
                path_str += f" ... → {self.path[-1]}"
            print(f"  Path          : {path_str}")


def reconstruct_path(came_from: Dict, start: Tuple, goal: Tuple) -> List[Position]:
    """CO2: Backtrack through came_from dict to get the full path."""
    path = []
    current = goal
    while current != start:
        path.append(Position(current[0], current[1]))
        current = came_from[current]
    path.append(Position(start[0], start[1]))
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────
# CO2: BFS — Breadth-First Search
# ─────────────────────────────────────────────────────────

def bfs(state: WarehouseState, start: Position, goal: Position,
        verbose: bool = True) -> SearchResult:
    """
    CO2: Breadth-First Search
    Strategy : Expand shallowest unexpanded node first (FIFO queue)
    Complete  : Yes (finite graph)
    Optimal   : Yes (for uniform step costs)
    Time      : O(b^d) — b=branching factor, d=depth of solution
    Space     : O(b^d) — stores all nodes at current depth
    """
    result = SearchResult("BFS")
    start_t = time.time()

    s = start.to_tuple()
    g = goal.to_tuple()

    frontier = deque([s])           # FIFO queue = open set
    came_from: Dict[Tuple, Optional[Tuple]] = {s: None}
    explored: Set[Tuple] = set()    # closed set

    if verbose:
        print(f"\n  CO2 ── BFS: {start} → {goal}")

    while frontier:
        result.open_set_sizes.append(len(frontier))
        current = frontier.popleft()
        result.nodes_expanded += 1
        result.explored_nodes.append(Position(current[0], current[1]))

        if current == g:
            result.found = True
            break

        explored.add(current)
        neighbors = []
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            npos = apply_action(Position(current[0], current[1]), action)
            nb = npos.to_tuple()
            if state.is_valid(npos) and nb not in came_from:
                came_from[nb] = current
                frontier.append(nb)
                neighbors.append(nb)

        if verbose and result.nodes_expanded <= 5:
            print(f"    Expand {current} | frontier={len(frontier)} | "
                  f"explored={len(explored)}")

    if result.found:
        result.path = reconstruct_path(came_from, s, g)
        result.path_cost = len(result.path) - 1

    result.runtime_ms = (time.time() - start_t) * 1000
    # Memory estimate: each node stored as tuple (2 ints = ~56 bytes)
    result.memory_est_kb = (len(came_from) * 56) / 1024
    return result


# ─────────────────────────────────────────────────────────
# CO2: DFS — Depth-First Search
# ─────────────────────────────────────────────────────────

def dfs(state: WarehouseState, start: Position, goal: Position,
        verbose: bool = True) -> SearchResult:
    """
    CO2: Depth-First Search (iterative with stack to avoid recursion limit)
    Strategy : Expand deepest unexpanded node first (LIFO stack)
    Complete  : Yes (with cycle detection in finite graphs)
    Optimal   : No — may find non-optimal path
    Time      : O(b^m) — m = max depth
    Space     : O(b*m) — linear space (major advantage)
    """
    result = SearchResult("DFS")
    start_t = time.time()

    s = start.to_tuple()
    g = goal.to_tuple()

    stack = [s]                     # LIFO stack = open set
    came_from: Dict[Tuple, Optional[Tuple]] = {s: None}
    explored: Set[Tuple] = set()

    if verbose:
        print(f"\n  CO2 ── DFS: {start} → {goal}")

    while stack:
        result.open_set_sizes.append(len(stack))
        current = stack.pop()
        result.nodes_expanded += 1
        result.explored_nodes.append(Position(current[0], current[1]))

        if current == g:
            result.found = True
            break

        if current in explored:
            continue
        explored.add(current)

        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            npos = apply_action(Position(current[0], current[1]), action)
            nb = npos.to_tuple()
            if state.is_valid(npos) and nb not in explored:
                if nb not in came_from:
                    came_from[nb] = current
                stack.append(nb)

        if verbose and result.nodes_expanded <= 5:
            print(f"    Expand {current} | stack={len(stack)} | "
                  f"explored={len(explored)}")

    if result.found:
        result.path = reconstruct_path(came_from, s, g)
        result.path_cost = len(result.path) - 1

    result.runtime_ms = (time.time() - start_t) * 1000
    result.memory_est_kb = (len(came_from) * 56) / 1024
    return result


# ─────────────────────────────────────────────────────────
# CO2: UCS — Uniform Cost Search
# ─────────────────────────────────────────────────────────

def ucs(state: WarehouseState, start: Position, goal: Position,
        verbose: bool = True) -> SearchResult:
    """
    CO2: Uniform Cost Search
    Strategy : Expand least-cost node first (min-heap priority queue)
    Complete  : Yes
    Optimal   : Yes — always finds least-cost path
    Time      : O(b^(1+C*/ε)) — C*=optimal cost, ε=min edge cost
    Space     : Same as time
    Note      : UCS = A* with h(n) = 0 (no heuristic)
    """
    result = SearchResult("UCS")
    start_t = time.time()

    s = start.to_tuple()
    g = goal.to_tuple()

    # Priority queue: (cost, node)
    heap = [(0, s)]
    came_from: Dict[Tuple, Optional[Tuple]] = {s: None}
    cost_so_far: Dict[Tuple, int] = {s: 0}

    if verbose:
        print(f"\n  CO2 ── UCS: {start} → {goal}")

    while heap:
        result.open_set_sizes.append(len(heap))
        cost, current = heapq.heappop(heap)
        result.nodes_expanded += 1
        result.explored_nodes.append(Position(current[0], current[1]))

        if current == g:
            result.found = True
            break

        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            npos = apply_action(Position(current[0], current[1]), action)
            nb = npos.to_tuple()
            if state.is_valid(npos):
                new_cost = cost + 1  # uniform edge cost = 1
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    came_from[nb] = current
                    heapq.heappush(heap, (new_cost, nb))

        if verbose and result.nodes_expanded <= 5:
            print(f"    Expand {current} cost={cost} | heap={len(heap)}")

    if result.found:
        result.path = reconstruct_path(came_from, s, g)
        result.path_cost = cost_so_far.get(g, 0)

    result.runtime_ms = (time.time() - start_t) * 1000
    result.memory_est_kb = (len(cost_so_far) * 72) / 1024
    return result


# ─────────────────────────────────────────────────────────
# CO2: A* SEARCH
# ─────────────────────────────────────────────────────────

def astar(state: WarehouseState, start: Position, goal: Position,
          verbose: bool = True) -> SearchResult:
    """
    CO2: A* Search
    Strategy : f(n) = g(n) + h(n)
               g(n) = cost from start to n
               h(n) = Manhattan distance heuristic (admissible)
    Complete  : Yes
    Optimal   : Yes (with admissible & consistent heuristic)
    Time      : O(b^d) — but h(n) dramatically prunes the search
    Space     : O(b^d) — keeps all nodes in memory

    KEY INSIGHT:
      A* is BFS/UCS + heuristic guidance.
      The heuristic steers expansion toward goal → fewer nodes explored.
    """
    result = SearchResult("A*")
    start_t = time.time()

    s = start.to_tuple()
    g = goal.to_tuple()
    goal_pos = goal

    # Priority queue: (f_cost, g_cost, node)
    h0 = manhattan_heuristic(start, goal)
    heap = [(h0, 0, s)]
    came_from: Dict[Tuple, Optional[Tuple]] = {s: None}
    g_cost: Dict[Tuple, int] = {s: 0}

    if verbose:
        print(f"\n  CO2 ── A*: {start} → {goal}")
        print(f"    Initial h(start) = {h0} (Manhattan distance)")

    while heap:
        result.open_set_sizes.append(len(heap))
        f, gc, current = heapq.heappop(heap)
        result.nodes_expanded += 1
        cur_pos = Position(current[0], current[1])
        result.explored_nodes.append(cur_pos)

        if current == g:
            result.found = True
            break

        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            npos = apply_action(cur_pos, action)
            nb = npos.to_tuple()
            if state.is_valid(npos):
                new_g = gc + 1
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    came_from[nb] = current
                    h = manhattan_heuristic(npos, goal_pos)
                    heapq.heappush(heap, (new_g + h, new_g, nb))

        if verbose and result.nodes_expanded <= 5:
            h = manhattan_heuristic(cur_pos, goal_pos)
            print(f"    Expand {current} g={gc} h={h} f={gc+h} | heap={len(heap)}")

    if result.found:
        result.path = reconstruct_path(came_from, s, g)
        result.path_cost = g_cost.get(g, 0)

    result.runtime_ms = (time.time() - start_t) * 1000
    result.memory_est_kb = (len(g_cost) * 80) / 1024
    return result


# ─────────────────────────────────────────────────────────
# CO2: PATH DISPLAY ON GRID
# ─────────────────────────────────────────────────────────

def display_path_on_grid(state: WarehouseState, result: SearchResult,
                          start: Position, goal: Position):
    """
    CO2: Overlay the searched path on the warehouse grid.
    Symbols:
      * = Path cells
      @ = Explored (but not on path) cells
      X = Obstacle
      S = Start, G = Goal
    """
    if not result.found:
        print("  [!] No path found to display.")
        return

    path_set = {p.to_tuple() for p in result.path}
    explored_set = {p.to_tuple() for p in result.explored_nodes}

    print(f"\n  CO2 ── PATH ON GRID ({result.algorithm})")
    print(f"  Legend: S=Start  G=Goal  *=Path  @=Explored  X=Obstacle")
    print()

    border = "  +" + "─────" * state.grid_cols + "+"
    print(border)
    for r in range(state.grid_rows):
        row_str = "  |"
        for c in range(state.grid_cols):
            pos = (r, c)
            if pos == start.to_tuple():
                cell = " S  "
            elif pos == goal.to_tuple():
                cell = " G  "
            elif pos in state.obstacles:
                cell = " X  "
            elif pos in path_set:
                cell = " *  "
            elif pos in explored_set:
                cell = " @  "
            else:
                cell = " .  "
            row_str += cell
        row_str += "|"
        print(row_str)
    print(border)
    print(f"  Path: {' → '.join(str(p) for p in result.path)}")


# ─────────────────────────────────────────────────────────
# CO2: ALGORITHM COMPARISON TABLE
# ─────────────────────────────────────────────────────────

def compare_algorithms(results: List[SearchResult]):
    """
    CO2: Side-by-side comparison of all search algorithms.
    Demonstrates the academic comparison: BFS vs A* etc.
    """
    print("\n" + "="*65)
    print("  CO2 ── ALGORITHM COMPARISON TABLE")
    print("="*65)
    print(f"  {'Algorithm':<12} {'Nodes':>8} {'Path Len':>9} {'Cost':>6} "
          f"{'Time(ms)':>10} {'Mem(KB)':>8} {'Found':>6}")
    print(f"  {'─'*11} {'─'*8} {'─'*9} {'─'*6} {'─'*10} {'─'*8} {'─'*6}")
    for r in results:
        found = "✓" if r.found else "✗"
        print(f"  {r.algorithm:<12} {r.nodes_expanded:>8} "
              f"{len(r.path):>9} {r.path_cost:>6} "
              f"{r.runtime_ms:>10.3f} {r.memory_est_kb:>8.2f} {found:>6}")

    print(f"\n  KEY INSIGHT:")
    astar_r = next((r for r in results if r.algorithm == "A*"), None)
    bfs_r   = next((r for r in results if r.algorithm == "BFS"), None)
    if astar_r and bfs_r and astar_r.found and bfs_r.found:
        savings = bfs_r.nodes_expanded - astar_r.nodes_expanded
        pct = (savings / max(bfs_r.nodes_expanded, 1)) * 100
        print(f"  A* explored {savings} fewer nodes than BFS ({pct:.1f}% reduction)")
        print(f"  This is the power of the Manhattan heuristic h(n)!")
    print()
