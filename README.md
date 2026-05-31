# Warehouse Robot Task Scheduler — AI Techniques

> A complete console-based AI project for academic study, mapping every feature to a Course Outcome (CO1–CO6).

---

## Project Overview

This system simulates a warehouse with multiple robots that:
- Pick and deliver packages across a grid
- Avoid static obstacles (shelves, walls)
- Manage battery levels intelligently
- Schedule tasks using Constraint Satisfaction
- Make decisions under sensor uncertainty using Bayes' Rule

---

## File Structure

```
warehouse_robot/
│
├── main.py                    ← Entry point; menu system
│
└── modules/
    ├── __init__.py
    ├── co1_environment.py     ← CO1: PEAS, State, Graph, Dataclasses
    ├── co2_search.py          ← CO2: BFS, DFS, UCS, A*
    ├── co3_csp.py             ← CO3: CSP, Backtracking, MRV, LCV
    ├── co4_utility.py         ← CO4: Utility, Multi-agent, Minimax
    ├── co5_probabilistic.py   ← CO5: Bayes Rule, Posterior, HMM
    ├── co6_hybrid.py          ← CO6: Hybrid pipeline, Traces
    └── grid_display.py        ← Grid rendering & user customization
```

---

## How to Run

**Requirements:** Python 3.8+ (no external libraries needed)

```bash
cd warehouse_robot
python main.py
```

You will see the warehouse grid, then a menu:

```
[1] CO1 — Environment & State
[2] CO2 — Search Algorithms
[3] CO3 — CSP Task Scheduling
[4] CO4 — Utility & Multi-Agent
[5] CO5 — Probabilistic Reasoning
[6] CO6 — Hybrid AI Pipeline
[7] Customize Warehouse Grid
[8] Show Grid
[9] CO Mapping Summary
[0] Exit
```

---

## Default Warehouse Grid

```
+─────────────────────────────+
| R1 | .  | .  | X  | .  | T1 |  row 0
| .  | .  | .  | X  | .  | .  |  row 1
| .  | X  | .  | .  | T2 | .  |  row 2
| .  | .  | .  | X  | .  | .  |  row 3
| R2 | .  | .  | .  | .  | D  |  row 4
+─────────────────────────────+
```

**Legend:** R1/R2=Robots | T1/T2=Tasks | X=Obstacle | D=Delivery | .=Empty

---

## CO-Wise Explanation

### CO1 — Problem Formulation

**File:** `modules/co1_environment.py`

| Concept | Implementation |
|---------|---------------|
| PEAS Model | `display_peas_model()` function |
| Environment Types | `display_environment_types()` |
| State | `RobotState` dataclass (position, battery, task) |
| Transition Model | `apply_action(pos, action) → Position` |
| Cost Function | `action_cost(action) → int` |
| Graph | `build_adjacency_graph(state) → Dict` |

**Key output:** Trace logs, state transitions, graph stats.

---

### CO2 — Search Algorithms

**File:** `modules/co2_search.py`

| Algorithm | Strategy | Optimal? | Nodes (example) |
|-----------|----------|----------|-----------------|
| BFS | FIFO queue | Yes | 26 |
| DFS | LIFO stack | No | 22 |
| UCS | Min-cost heap | Yes | 25 |
| A* | f=g+h, Manhattan | Yes | 15 ✓ |

**Heuristic:** Manhattan distance h(n) = |Δrow| + |Δcol|
- Admissible: never overestimates
- A* explores 42% fewer nodes than BFS (example run)

**Output:** Path on ASCII grid, comparison table, timing.

---

### CO3 — CSP Scheduling

**File:** `modules/co3_csp.py`

| Component | Description |
|-----------|-------------|
| Variables | Tasks (T1, T2, ...) |
| Domains | Robots (R1, R2, ...) |
| Constraint C1 | Battery ≥ 2 × task distance |
| Constraint C3 | No robot handles 2 tasks at once |
| MRV | Pick task with fewest legal robots |
| LCV | Try robot that leaves most options open |
| Forward Check | Prune domains after each assignment |

**Failure messages** are always explained, e.g.:
> `[C1-FAIL] T2 cannot be assigned to R2: battery=30% requires ≥40%`

---

### CO4 — Utility & Multi-Agent

**File:** `modules/co4_utility.py`

**Utility function:**
```
U(robot, task) = 0.30×battery + 0.40×proximity + 0.10×task_cost + 0.20×priority
```

**Narrow aisle conflict:** Two robots computed urgency scores; higher urgency gets priority.

**Minimax intuition:** Explains adversarial decision trees with Alpha-Beta pruning.

**Bounded Rationality:** Depth-limited search + evaluation function (H-Minimax).

---

### CO5 — Probabilistic Reasoning

**File:** `modules/co5_probabilistic.py`

**Bayes' Rule applied:**
```
P(Obstacle | Sensor=True) = P(Sensor=True | Obstacle) × P(Obstacle) / P(Sensor=True)
```

| Scenario | Prior | Sensor | Posterior | Decision |
|----------|-------|--------|-----------|----------|
| 1 | 0.30 | FIRED | 0.7941 | REROUTE |
| 2 | 0.50 | silent | 0.1000 | PROCEED |

**Sequential updating:** Belief refined over multiple sensor readings.

**Concepts explained:** Bayesian Networks, HMM, Expected Utility.

---

### CO6 — Hybrid AI Architecture

**File:** `modules/co6_hybrid.py`

Full pipeline:
1. **CO3 CSP** → Schedule tasks to robots
2. **CO5 Bayesian** → Check each path for uncertain obstacles
3. **CO2 A*** → Plan optimal paths (reroute if needed)
4. **CO4 Utility** → Resolve multi-robot conflicts

**Explainable trace:** Every decision logged with module + reason + confidence.

**Critical analysis:** Heuristic bias, uncertainty limits, scalability discussion.

---

## Customization

Choose option `[7]` to:
- Move robots to new coordinates
- Change robot battery levels
- Modify task pickup/dropoff locations
- Set task priorities (1=low, 2=medium, 3=high)
- Add or remove obstacles interactively

Grid re-renders after every change.

---

## Sample Viva Questions & Answers

**Q: Why is A* better than BFS?**
A: A* uses a heuristic h(n) to guide expansion toward the goal, reducing unnecessary node exploration. In our demo, A* expanded 42% fewer nodes than BFS while still finding the optimal path.

**Q: What is MRV in CSP?**
A: Minimum Remaining Values — choose the variable (task) with the fewest legal values (robots) remaining. This detects failures early, pruning the search tree.

**Q: Why use Bayesian reasoning for sensors?**
A: Sensors are noisy. Bayes' Rule combines prior belief with sensor evidence to compute a posterior probability, enabling rational decisions under uncertainty (reroute only if P > threshold).

**Q: How does minimax apply to robots?**
A: Two competing robots (for a charging station or aisle) can be modeled as a 2-player game: one maximizes its outcome, the other minimizes it. Alpha-beta pruning makes this efficient.

**Q: What is the scalability limitation?**
A: CSP backtracking is O(d^n) — exponential in the number of tasks. For 100+ tasks, genetic algorithms or reinforcement learning would be more practical.

---

## Academic References

- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.)
  - Ch. 2: Intelligent Agents (CO1)
  - Ch. 3-4: Search (CO2)
  - Ch. 6: CSP (CO3)
  - Ch. 5: Adversarial Search (CO4)
  - Ch. 12-13: Probabilistic Reasoning (CO5)
