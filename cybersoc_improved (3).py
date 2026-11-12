"""
CyberSOC prototype - five cooperating agents on simulated data.

Run:
    python cybersoc_improved.py

Python 3.8+, standard library only.

IMPORTANT:
All data in this prototype is synthetic. Detection metrics are measured on a
repeatable synthetic benchmark and are not evidence of real-world accuracy.
"""

import math
import heapq
import random
from collections import defaultdict

# ------------------------- Simulated environment -------------------------

CRITICALITY = {
    "web01": 2,
    "mail01": 3,
    "wks-hr1": 1,
    "wks-fin2": 2,
    "fileserver": 5,
    "backup01": 6,
    "ad-dc": 9,
    "db-prod": 10,
}

# (source, destination): probability of propagation in one simulation step
LINKS = {
    ("mail01", "wks-hr1"): 0.6,
    ("wks-hr1", "wks-fin2"): 0.5,
    ("wks-hr1", "fileserver"): 0.4,
    ("wks-fin2", "fileserver"): 0.5,
    ("wks-fin2", "ad-dc"): 0.25,
    ("fileserver", "ad-dc"): 0.35,
    ("fileserver", "backup01"): 0.3,
    ("ad-dc", "db-prod"): 0.7,
    ("ad-dc", "backup01"): 0.6,
    ("web01", "db-prod"): 0.1,
    ("mail01", "ad-dc"): 0.1,
}

# (minute, host, event)
LOGS = [
    (0, "mail01", "phishing_delivered"),
    (4, "wks-hr1", "macro_exec"),
    (9, "wks-hr1", "c2_beacon"),
    (15, "wks-hr1", "cred_dump"),
    (22, "wks-fin2", "remote_logon_anomaly"),
    (30, "wks-fin2", "c2_beacon"),
]

STAGES = {
    "phishing_delivered": ("Initial Access Attempt", 2),
    "macro_exec": ("Execution", 5),
    "c2_beacon": ("Command & Control", 6),
    "cred_dump": ("Credential Access", 8),
    "remote_logon_anomaly": ("Lateral Movement", 7),
}

# Ground-truth labels for this synthetic demonstration only.
# mail01 received a phishing message but is not labeled as compromised.
GROUND_TRUTH_COMPROMISED = {"wks-hr1", "wks-fin2"}


# ------------------------- Agent 1: Detection -------------------------

def detection_agent(logs, threshold=7):
    """Score hosts from event severity and return hosts above the threshold."""
    scores = defaultdict(int)
    for _, host, event in logs:
        if event not in STAGES:
            continue
        scores[host] += STAGES[event][1]
    alerts = {host: score for host, score in scores.items() if score >= threshold}
    return alerts, dict(scores)


def evaluate_detection(predicted_hosts, actual_hosts, all_hosts):
    """Calculate standard classification metrics against synthetic labels."""
    predicted = set(predicted_hosts)
    actual = set(actual_hosts)
    universe = set(all_hosts)

    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(universe - (predicted | actual))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "total": len(universe),
        "actual_positive": len(actual & universe),
        "actual_negative": len(universe - actual),
    }


def build_synthetic_evaluation_set(host_count=200, positive_rate=0.25, seed=2026):
    """Create a reproducible benchmark with independent host labels and logs.

    The benchmark deliberately includes ambiguous benign events and compromised
    hosts with limited evidence, so the rule-based detector can make both false
    positive and false negative predictions. This is a teaching benchmark, not
    a substitute for real, independently collected security telemetry.
    """
    if host_count <= 0:
        raise ValueError("host_count must be greater than zero")
    if not 0 < positive_rate < 1:
        raise ValueError("positive_rate must be between 0 and 1")

    rng = random.Random(seed)
    positive_count = max(1, min(host_count - 1, round(host_count * positive_rate)))
    labels = [True] * positive_count + [False] * (host_count - positive_count)
    rng.shuffle(labels)

    benign_weights = (
        ("phishing_delivered", 5),
        ("macro_exec", 2),
        ("c2_beacon", 1),
        ("cred_dump", 1),
        ("remote_logon_anomaly", 3),
    )
    compromised_weights = (
        ("phishing_delivered", 1),
        ("macro_exec", 3),
        ("c2_beacon", 4),
        ("cred_dump", 3),
        ("remote_logon_anomaly", 3),
    )

    evaluation_logs = []
    actual_hosts = set()
    for index, is_compromised in enumerate(labels):
        host = "eval-host-{0:03d}".format(index + 1)
        if is_compromised:
            actual_hosts.add(host)
            event_count = rng.randint(1, 4)
            weighted_events = compromised_weights
        else:
            event_count = rng.randint(0, 2)
            weighted_events = benign_weights

        events, weights = zip(*weighted_events)
        for event_index in range(event_count):
            event = rng.choices(events, weights=weights, k=1)[0]
            minute = event_index * rng.randint(1, 8) + rng.randint(0, 3)
            evaluation_logs.append((minute, host, event))

    all_hosts = {"eval-host-{0:03d}".format(i + 1) for i in range(host_count)}
    return evaluation_logs, actual_hosts, all_hosts


# ------------------------- Agent 2: Past (forensics) -------------------------

def past_agent(logs, alerts):
    """
    Reconstruct the full supplied event sequence, not just alerted hosts.
    This preserves the initial phishing event that precedes compromise.
    """
    timeline = [
        (minute, host, STAGES[event][0])
        for minute, host, event in sorted(logs)
        if event in STAGES
    ]

    if not timeline:
        return [], None, 0, None

    entry_point = timeline[0][1]
    confirmed_compromise_hosts = set(alerts)
    first_compromised = next(
        (host for _, host, _ in timeline if host in confirmed_compromise_hosts),
        None,
    )
    duration = timeline[-1][0] - timeline[0][0]

    return timeline, entry_point, duration, first_compromised


# ------------------------- Agent 3: Present -------------------------

def present_agent(alerts):
    compromised = set(alerts)
    impact = sum(CRITICALITY.get(host, 0) for host in compromised)
    level = "CRITICAL" if impact >= 5 or len(compromised) > 1 else "HIGH"
    return compromised, impact, level


# ------------------------- Agent 4: Future (propagation) -------------------------

def simulate(compromised, links, steps=5, runs=5000, seed=7):
    """
    Monte Carlo propagation estimate.

    A local RNG makes each run reproducible. Use the same seed when comparing
    response scenarios to reduce noise from random-number variation.
    """
    rng = random.Random(seed)
    hits = defaultdict(int)

    for _ in range(runs):
        owned = set(compromised)

        for _ in range(steps):
            new_hosts = {
                destination
                for (source, destination), probability in links.items()
                if source in owned
                and destination not in owned
                and rng.random() < probability
            }
            if not new_hosts:
                break
            owned.update(new_hosts)

        for host in owned:
            hits[host] += 1

    probabilities = {
        host: hits[host] / runs
        for host in CRITICALITY
    }

    expected_impact = sum(
        probabilities[host] * CRITICALITY[host]
        for host in probabilities
        if host not in compromised
    )
    return probabilities, expected_impact


def likeliest_path(compromised, links, target):
    """Find the highest-probability path using Dijkstra on -log(probability)."""
    queue = [(0.0, host, [host]) for host in compromised]
    heapq.heapify(queue)
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)

        if node == target:
            return path, math.exp(-cost)

        if node in visited:
            continue
        visited.add(node)

        for (source, destination), probability in links.items():
            if source == node and destination not in visited and probability > 0:
                queue.append((
                    cost - math.log(probability),
                    destination,
                    path + [destination],
                ))
        heapq.heapify(queue)

    return None, 0.0


# ------------------------- Agent 5: What-If engine -------------------------

ACTIONS = {
    "Isolate fileserver": (
        4,
        lambda links: {
            edge: probability
            for edge, probability in links.items()
            if "fileserver" not in edge
        },
    ),
    "Isolate wks-fin2": (
        2,
        lambda links: {
            edge: probability
            for edge, probability in links.items()
            if "wks-fin2" not in edge
        },
    ),
    "Reset credentials + harden AD": (
        3,
        lambda links: {
            edge: (
                probability * 0.3
                if edge[1] == "ad-dc"
                else probability
            )
            for edge, probability in links.items()
        },
    ),
    "Patch & segment backup01": (
        2,
        lambda links: {
            edge: (
                probability * 0.2
                if edge[1] == "backup01"
                else probability
            )
            for edge, probability in links.items()
        },
    ),
}


def whatif_agent(compromised, base_risk, runs=5000, seed=7):
    """Compare response options using the same simulation seed."""
    results = []

    for name, (cost, transform_links) in ACTIONS.items():
        modified_links = transform_links(LINKS)
        _, scenario_risk = simulate(
            compromised,
            modified_links,
            runs=runs,
            seed=seed,
        )
        risk_reduction = base_risk - scenario_risk
        efficiency = risk_reduction / cost if cost else 0.0

        results.append({
            "name": name,
            "cost": cost,
            "risk": scenario_risk,
            "reduction": risk_reduction,
            "efficiency": efficiency,
        })

    results.sort(key=lambda item: item["efficiency"], reverse=True)
    return results


# ------------------------- Reporting -------------------------

def main():
    print("=== CyberSOC: Multi-Agent Cybersecurity Prototype ===")
    print("DEMO MODE: all events and labels are synthetic.\n")

    # Agent 1
    alerts, scores = detection_agent(LOGS)
    print("[1] THREAT DETECTION")
    print("Host risk scores:", scores)
    print("Detected compromised hosts:", alerts)

    # Evaluate the same detector on a larger, reproducible synthetic benchmark.
    evaluation_logs, evaluation_truth, evaluation_hosts = build_synthetic_evaluation_set()
    evaluation_alerts, _ = detection_agent(evaluation_logs)
    metrics = evaluate_detection(
        evaluation_alerts.keys(), evaluation_truth, evaluation_hosts
    )
    print("\n[DETECTION EVALUATION - SYNTHETIC BENCHMARK]")
    print("Hosts evaluated: {0}".format(metrics["total"]))
    print("Actual compromised hosts: {0}".format(metrics["actual_positive"]))
    print("Actual benign hosts: {0}".format(metrics["actual_negative"]))
    print(
        "Accuracy:          {0:.1%}\n"
        "Precision:         {1:.1%}\n"
        "Recall (sensitivity): {2:.1%}\n"
        "Specificity:       {3:.1%}\n"
        "F1-score:          {4:.1%}\n"
        "Balanced accuracy: {5:.1%}".format(
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["specificity"], metrics["f1"], metrics["balanced_accuracy"]
        )
    )
    print(
        "Confusion matrix counts: "
        "TP={TP}, FP={FP}, FN={FN}, TN={TN}".format(**metrics)
    )
    print("Benchmark seed: 2026 (results are reproducible)")
    print("Note: synthetic benchmark results do not represent real-world accuracy.")

    # Agent 2
    timeline, entry_point, duration, first_compromised = past_agent(LOGS, alerts)
    print("\n[2] PAST - ATTACK RECONSTRUCTION")
    print("Initial attack entry point:", entry_point)
    print("First host identified as compromised:", first_compromised)
    print("Observed timeline duration:", duration, "minutes")
    for minute, host, stage in timeline:
        print("  t+{0:>2}m  {1:<10} {2}".format(minute, host, stage))

    # Agent 3
    compromised, impact, severity = present_agent(alerts)
    print("\n[3] PRESENT - CURRENT SECURITY STATE")
    print("Compromised hosts:", sorted(compromised))
    print("Impact score:", impact)
    print("Severity:", severity)

    # Agent 4
    probabilities, baseline_risk = simulate(compromised, LINKS)
    print("\n[4] FUTURE - PREDICTED PROPAGATION")
    print("Estimated probability over the next 5 simulation steps:")
    for host, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if host not in compromised:
            print("  {0:<10} {1:5.1%}".format(host, probability))

    path, path_probability = likeliest_path(compromised, LINKS, "db-prod")
    if path:
        print(
            "Most likely modeled path to db-prod: {0} "
            "(path probability: {1:.1%})".format(
                " -> ".join(path),
                path_probability,
            )
        )
    else:
        print("No modeled path to db-prod was found.")

    print("Expected additional impact score: {:.2f}".format(baseline_risk))

    # Agent 5
    print("\n[5] WHAT-IF - RESPONSE SCENARIO COMPARISON")
    response_results = whatif_agent(compromised, baseline_risk)

    for result in response_results:
        if baseline_risk > 0:
            reduction_percent = result["reduction"] / baseline_risk
        else:
            reduction_percent = 0.0

        print(
            "  {name:<32} cost={cost}  risk={risk:.2f}  "
            "reduction={reduction:.2f} ({percent:.1%})".format(
                name=result["name"],
                cost=result["cost"],
                risk=result["risk"],
                reduction=result["reduction"],
                percent=reduction_percent,
            )
        )

    if response_results:
        print("\nRecommended modeled option:", response_results[0]["name"])

    print("\nEVALUATION NOTE:")
    print(
        "The benchmark is generated data, not independently collected security traffic."
    )
    print(
        "For a credible real-world evaluation, test on a separate labeled dataset "
        "collected from representative environments and not used for tuning."
    )


if __name__ == "__main__":
    main()
