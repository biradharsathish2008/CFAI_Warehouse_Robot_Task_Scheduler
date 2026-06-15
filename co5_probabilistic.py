"""
=============================================================
CO5: Probabilistic Reasoning & Uncertainty
=============================================================
Topics Covered:
  - Bayes' Rule
  - Bayesian obstacle detection (sensor uncertainty)
  - Posterior probability calculation
  - Robot rerouting decision under uncertainty
  - Bayesian Network intuition
  - HMM intuition for robot tracking
  - Expected Utility concept
=============================================================
"""

from typing import Dict, Tuple, List
from modules.co1_environment import Position, RobotState, WarehouseState


# ─────────────────────────────────────────────────────────
# CO5: BAYES' RULE
# ─────────────────────────────────────────────────────────

def display_bayes_rule():
    """
    CO5: Display and explain Bayes' Rule in the warehouse context.
    P(Obstacle|Sensor=True) = P(Sensor=True|Obstacle) * P(Obstacle) / P(Sensor=True)
    """
    print("\n" + "="*60)
    print("  CO5 ── BAYES' RULE")
    print("="*60)
    print("""
  Bayes' Rule:
  ────────────────────────────────────────────────────────
  P(H | E) = P(E | H) × P(H)
             ─────────────────
                  P(E)

  Where:
    H = Hypothesis  (e.g., "Obstacle is present")
    E = Evidence    (e.g., "Sensor fired = True")
    P(H)      = Prior probability of obstacle
    P(E | H)  = Likelihood: sensor detection rate given obstacle
    P(E)      = Total probability of sensor firing
    P(H | E)  = Posterior: updated belief after sensor reading

  Expanding P(E):
    P(E) = P(E|H)×P(H) + P(E|¬H)×P(¬H)
         = (true_positive_rate × prior) +
           (false_positive_rate × (1 - prior))
    """)


# ─────────────────────────────────────────────────────────
# CO5: SENSOR MODEL
# ─────────────────────────────────────────────────────────

class SensorModel:
    """
    CO5: Probabilistic sensor model for obstacle detection.
    Sensor is not perfect:
      - True Positive Rate (TPR) : P(sensor=True | obstacle=True)
      - False Positive Rate (FPR): P(sensor=True | obstacle=False)
    """

    def __init__(self, true_positive_rate: float = 0.90,
                 false_positive_rate: float = 0.10):
        """
        Default sensor:
          90% chance of detecting an obstacle that IS there (TPR)
          10% chance of falsely reporting an obstacle that is NOT there (FPR)
        """
        self.tpr = true_positive_rate   # P(E=True | Obstacle)
        self.fpr = false_positive_rate  # P(E=True | No Obstacle)

    def posterior(self, prior: float, sensor_fired: bool) -> float:
        """
        CO5: Calculate P(Obstacle | Sensor Reading) using Bayes' Rule.

        Args:
            prior       : P(Obstacle) — prior belief
            sensor_fired: True if sensor detected obstacle

        Returns:
            posterior   : P(Obstacle | Evidence)
        """
        if sensor_fired:
            # P(E=True | Obstacle) × P(Obstacle)
            p_e_given_h    = self.tpr
            p_e_given_not_h = self.fpr
        else:
            # P(E=False | Obstacle) × P(Obstacle)
            p_e_given_h    = 1.0 - self.tpr
            p_e_given_not_h = 1.0 - self.fpr

        # Total probability of evidence
        p_e = p_e_given_h * prior + p_e_given_not_h * (1.0 - prior)

        if p_e == 0:
            return 0.0

        # Bayes' Rule
        posterior = (p_e_given_h * prior) / p_e
        return round(posterior, 4)


# ─────────────────────────────────────────────────────────
# CO5: OBSTACLE DETECTION SCENARIO
# ─────────────────────────────────────────────────────────

def bayesian_obstacle_decision(robot: RobotState, target_cell: Position,
                                sensor_model: SensorModel,
                                prior: float,
                                sensor_fired: bool,
                                reroute_threshold: float = 0.60) -> bool:
    """
    CO5: Make rerouting decision using Bayesian posterior.

    Scenario:
      Robot approaches a cell. Sensor fires (or not).
      Calculate P(obstacle | sensor reading).
      If posterior > threshold → reroute.
      Otherwise → proceed (accept risk).

    Returns: True if robot should reroute, False if proceed.
    """
    posterior = sensor_model.posterior(prior, sensor_fired)

    print(f"\n  CO5 ── BAYESIAN OBSTACLE DETECTION")
    print(f"  {'─'*50}")
    print(f"  Robot           : {robot.robot_id} at {robot.position}")
    print(f"  Target Cell     : {target_cell}")
    print(f"  Prior P(Obs)    : {prior:.2f}")
    print(f"  Sensor fired    : {'YES' if sensor_fired else 'NO'}")
    print(f"  TPR             : {sensor_model.tpr:.2f}  (true detection rate)")
    print(f"  FPR             : {sensor_model.fpr:.2f}  (false alarm rate)")
    print()
    print(f"  Bayes Calculation:")
    print(f"  ─────────────────")
    if sensor_fired:
        p_e_h  = sensor_model.tpr
        p_e_nh = sensor_model.fpr
    else:
        p_e_h  = 1 - sensor_model.tpr
        p_e_nh = 1 - sensor_model.fpr
    p_e = p_e_h * prior + p_e_nh * (1 - prior)
    print(f"  P(E|H)          : {p_e_h:.2f}")
    print(f"  P(E|¬H)         : {p_e_nh:.2f}")
    print(f"  P(E) [total]    : {p_e:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  P(Obs|Evidence) : {posterior:.4f}  ← Posterior Belief")
    print(f"  Reroute if      : > {reroute_threshold:.2f}")
    print()

    should_reroute = posterior > reroute_threshold

    if should_reroute:
        print(f"  DECISION: 🔄 REROUTE — posterior ({posterior:.4f}) > "
              f"threshold ({reroute_threshold:.2f})")
        print(f"  REASON  : High probability of real obstacle. "
              f"Safety first — find alternate path.")
    else:
        print(f"  DECISION: ✓ PROCEED — posterior ({posterior:.4f}) ≤ "
              f"threshold ({reroute_threshold:.2f})")
        print(f"  REASON  : Low-to-moderate probability. "
              f"Risk acceptable — proceed with caution.")

    return should_reroute


# ─────────────────────────────────────────────────────────
# CO5: SEQUENTIAL BAYESIAN UPDATING
# ─────────────────────────────────────────────────────────

def sequential_bayesian_update(sensor_model: SensorModel,
                                prior: float,
                                sensor_readings: List[bool]) -> List[float]:
    """
    CO5: Update belief sequentially as sensor readings arrive.
    Each reading updates the prior for the next step.
    Demonstrates: belief becomes more accurate with more evidence.
    """
    print(f"\n  CO5 ── SEQUENTIAL BAYESIAN UPDATING")
    print(f"  {'─'*50}")
    print(f"  Initial prior P(Obstacle) = {prior:.2f}")
    print(f"  Sensor readings sequence  : {sensor_readings}")
    print()

    beliefs = [prior]
    current = prior

    for i, reading in enumerate(sensor_readings):
        updated = sensor_model.posterior(current, reading)
        beliefs.append(updated)
        direction = "↑" if updated > current else "↓"
        print(f"  Reading {i+1}: sensor={'FIRED' if reading else 'quiet':7s} | "
              f"P(Obs) = {current:.4f} {direction} {updated:.4f}")
        current = updated

    print(f"\n  Final belief: P(Obstacle) = {current:.4f}")
    print(f"  {'STRONG evidence of obstacle' if current > 0.8 else 'Uncertain' if current > 0.4 else 'Likely no obstacle'}")
    return beliefs


# ─────────────────────────────────────────────────────────
# CO5: EDUCATIONAL EXPLANATIONS
# ─────────────────────────────────────────────────────────

def explain_bayesian_network():
    """CO5: Explain Bayesian Network intuition."""
    print("\n" + "─"*60)
    print("  CO5 ── BAYESIAN NETWORK INTUITION")
    print("─"*60)
    print("""
  A Bayesian Network is a Directed Acyclic Graph (DAG) where:
  • Nodes = Random Variables
  • Edges = Conditional Dependencies
  • Each node has a Conditional Probability Table (CPT)

  Warehouse Example:
  ──────────────────
    [Shelf_Collapsed] ──→ [Path_Blocked] ──→ [Sensor_Fires]
         ↓                     ↓
    [Robot_Delayed]    [Task_Incomplete]

  • P(Sensor_Fires | Path_Blocked=True)  = 0.90  [TPR]
  • P(Sensor_Fires | Path_Blocked=False) = 0.10  [FPR]
  • P(Path_Blocked | Shelf_Collapsed)    = 0.95
  • P(Path_Blocked | ¬Shelf_Collapsed)   = 0.05

  Inference: Given Sensor fires, what is P(Shelf_Collapsed)?
  → Chain through the network using Bayes' Rule at each node.
    """)


def explain_hmm():
    """CO5: Explain Hidden Markov Model intuition for robot tracking."""
    print("\n" + "─"*60)
    print("  CO5 ── HMM INTUITION FOR ROBOT TRACKING")
    print("─"*60)
    print("""
  Hidden Markov Model (HMM):
  ─────────────────────────
  • Hidden states  : Robot's TRUE position (unknown)
  • Observations   : Sensor readings (noisy position estimates)
  • Transition model: P(pos_t | pos_{t-1}) — how robot moves
  • Emission model : P(sensor_t | pos_t) — sensor accuracy

  Why "hidden"?
  → We can't directly observe the robot's exact position.
  → GPS/sensors give noisy readings.
  → HMM estimates the MOST LIKELY sequence of true positions.

  Algorithms for HMM:
    • Forward Algorithm    : P(observations) — likelihood
    • Viterbi Algorithm    : Most likely state sequence
    • Baum-Welch           : Learn HMM parameters from data

  Warehouse Application:
  → Track robot position over time despite sensor noise.
  → Estimate where a robot ACTUALLY is vs where sensor says.
  → Detect anomalies (robot took wrong path, collision).
    """)


def explain_expected_utility():
    """CO5: Explain Expected Utility concept."""
    print("\n" + "─"*60)
    print("  CO5 ── EXPECTED UTILITY CONCEPT")
    print("─"*60)
    print("""
  Expected Utility EU(action) = Σ P(outcome_i | action) × U(outcome_i)

  Example: Robot deciding whether to proceed through uncertain path.

  Action: PROCEED
    P(obstacle=True | sensor)  = 0.70 → outcome: collision, U = -100
    P(obstacle=False | sensor) = 0.30 → outcome: success,   U = +50
    EU(PROCEED) = 0.70 × (-100) + 0.30 × 50 = -70 + 15 = -55

  Action: REROUTE
    P(reroute succeeds)        = 0.95 → U = +30 (longer but safe)
    P(reroute also blocked)    = 0.05 → U = -20
    EU(REROUTE) = 0.95 × 30 + 0.05 × (-20) = 28.5 - 1 = +27.5

  DECISION: REROUTE has higher expected utility (+27.5 > -55)
  → Rational agent picks action with highest EU!
    """)


# ─────────────────────────────────────────────────────────
# CO5: RUN CO5 DEMO
# ─────────────────────────────────────────────────────────

def run_co5_demo(state: WarehouseState):
    """CO5: Run all probabilistic reasoning demonstrations."""

    # Display Bayes' Rule
    display_bayes_rule()

    sensor = SensorModel(true_positive_rate=0.90, false_positive_rate=0.10)

    # Pick first robot
    robot = list(state.robots.values())[0]
    target = Position(robot.position.row, robot.position.col + 2)

    # Scenario 1: Sensor fired
    print("\n  ── SCENARIO 1: Sensor fires (possible obstacle ahead) ──")
    bayesian_obstacle_decision(
        robot, target, sensor,
        prior=0.30,          # 30% prior chance of obstacle
        sensor_fired=True,   # sensor says "obstacle!"
        reroute_threshold=0.60
    )

    # Scenario 2: Sensor silent
    print("\n  ── SCENARIO 2: Sensor silent (no obstacle detected) ──")
    bayesian_obstacle_decision(
        robot, target, sensor,
        prior=0.50,
        sensor_fired=False,
        reroute_threshold=0.60
    )

    # Sequential updating
    sequential_bayesian_update(
        sensor, prior=0.30,
        sensor_readings=[True, True, False, True]
    )

    # Educational explanations
    explain_bayesian_network()
    explain_hmm()
    explain_expected_utility()
