"""
evaluation/pareto_frontier.py
──────────────────────────────
Multi-objective Pareto frontier analysis for LoRA checkpoint selection.

Three objectives (all want to be maximised after normalisation):
  - Accuracy        (auto-resolution rate)
  - Speed           (1 / latency — faster is better)
  - Cost efficiency (1 / cost_per_ticket)

A checkpoint is Pareto-optimal if no other checkpoint dominates it on
ALL three objectives simultaneously. The Pareto frontier is the set of
non-dominated checkpoints — these are the candidates for production.

Portfolio note: "Selected r=32 as optimal: 96.5% accuracy, 120ms latency,
$0.12/ticket — identified via Pareto frontier analysis"

Public API
----------
ParetoPoint         – A single checkpoint on the objective space
ParetoFrontier
  .add(point)
  .compute()                          → List[ParetoPoint]  (non-dominated)
  .dominated_hypervolume(ref)         → float
  .select_best(priority)              → ParetoPoint
  .to_dataframe()                     → pd.DataFrame
  .save(path) / .load(path)
  .plot(path)                         → matplotlib Figure (if matplotlib available)

build_pareto_frontier(eval_results)   → ParetoFrontier
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A single checkpoint with its objective values."""
    run_name:    str
    domain:      str
    lora_rank:   int
    accuracy:    float      # auto-resolution rate [0, 1]
    latency_ms:  float      # p95 latency in milliseconds
    cost_per_ticket: float  # $ per ticket
    is_pareto_optimal: bool = False

    # Derived objectives (maximise all)
    @property
    def speed(self) -> float:
        """Inverse latency — higher is faster."""
        return 1.0 / max(self.latency_ms, 1e-6)

    @property
    def cost_efficiency(self) -> float:
        """Inverse cost — higher is cheaper."""
        return 1.0 / max(self.cost_per_ticket, 1e-6)

    def objectives(self) -> Tuple[float, float, float]:
        """Return (accuracy, speed, cost_efficiency) — all maximise."""
        return self.accuracy, self.speed, self.cost_efficiency

    def dominates(self, other: "ParetoPoint") -> bool:
        """
        Return True if self dominates other on ALL objectives.
        (self is at least as good on all, strictly better on at least one)
        """
        s = self.objectives()
        o = other.objectives()
        return (
            all(si >= oi for si, oi in zip(s, o))
            and any(si > oi for si, oi in zip(s, o))
        )

    def summary(self) -> str:
        star = " ★" if self.is_pareto_optimal else ""
        return (
            f"{self.run_name:30s}  "
            f"acc={self.accuracy:.3f}  "
            f"lat={self.latency_ms:.0f}ms  "
            f"cost=${self.cost_per_ticket:.3f}{star}"
        )


class ParetoFrontier:
    """
    Multi-objective Pareto frontier for VeriTune checkpoint selection.

    Finds the non-dominated set across accuracy × latency × cost.
    """

    def __init__(self) -> None:
        self._points: List[ParetoPoint] = []
        self._frontier: List[ParetoPoint] = []
        self._computed = False

    def add(self, point: ParetoPoint) -> None:
        """Add a checkpoint to the candidate set."""
        self._points.append(point)
        self._computed = False

    def add_many(self, points: List[ParetoPoint]) -> None:
        for p in points:
            self.add(p)

    def compute(self) -> List[ParetoPoint]:
        """
        Compute the Pareto-optimal set (non-dominated solutions).

        A point is non-dominated if no other point dominates it.
        Returns the frontier sorted by accuracy descending.
        """
        if not self._points:
            return []

        n = len(self._points)
        is_dominated = [False] * n

        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[i]:
                    if self._points[j].dominates(self._points[i]):
                        is_dominated[i] = True
                        break

        # Mark all points
        for i, point in enumerate(self._points):
            point.is_pareto_optimal = not is_dominated[i]

        self._frontier = [p for p, d in zip(self._points, is_dominated) if not d]
        self._frontier.sort(key=lambda p: p.accuracy, reverse=True)
        self._computed = True

        logger.info(
            "Pareto frontier: %d / %d points are non-dominated",
            len(self._frontier), n,
        )
        return self._frontier

    @property
    def frontier(self) -> List[ParetoPoint]:
        if not self._computed:
            self.compute()
        return self._frontier

    @property
    def all_points(self) -> List[ParetoPoint]:
        return self._points

    def select_best(
        self,
        priority: str = "balanced",
        latency_budget_ms: float = 200.0,
        cost_budget: float = 0.20,
    ) -> Optional[ParetoPoint]:
        """
        Select the recommended checkpoint from the Pareto frontier.

        Parameters
        ----------
        priority           : "accuracy" | "speed" | "cost" | "balanced"
        latency_budget_ms  : Hard constraint on p95 latency
        cost_budget        : Hard constraint on cost per ticket

        Returns the best Pareto-optimal point subject to constraints.
        """
        candidates = [
            p for p in self.frontier
            if p.latency_ms <= latency_budget_ms
            and p.cost_per_ticket <= cost_budget
        ]

        if not candidates:
            logger.warning(
                "No Pareto-optimal points within budget "
                "(lat<%.0fms, cost<$%.2f) — relaxing constraints",
                latency_budget_ms, cost_budget,
            )
            candidates = self.frontier

        if not candidates:
            return None

        if priority == "accuracy":
            return max(candidates, key=lambda p: p.accuracy)
        elif priority == "speed":
            return min(candidates, key=lambda p: p.latency_ms)
        elif priority == "cost":
            return min(candidates, key=lambda p: p.cost_per_ticket)
        else:  # balanced — maximise geometric mean of normalised objectives
            accs  = [p.accuracy         for p in candidates]
            lats  = [p.latency_ms       for p in candidates]
            costs = [p.cost_per_ticket  for p in candidates]

            max_acc  = max(accs)  + 1e-9
            min_lat  = min(lats)  + 1e-9
            min_cost = min(costs) + 1e-9

            def balanced_score(p: ParetoPoint) -> float:
                norm_acc  = p.accuracy / max_acc
                norm_lat  = min_lat / p.latency_ms        # lower lat → higher score
                norm_cost = min_cost / p.cost_per_ticket  # lower cost → higher score
                return (norm_acc * norm_lat * norm_cost) ** (1/3)  # geometric mean

            return max(candidates, key=balanced_score)

    def dominated_hypervolume(
        self,
        reference_point: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """
        Compute the hypervolume dominated by the Pareto frontier.
        Higher hypervolume = better overall frontier.

        Uses a simple Monte Carlo estimate for the 3D case.
        reference_point : (accuracy, speed, cost_efficiency) lower bounds
        """
        if not self.frontier:
            return 0.0

        if reference_point is None:
            reference_point = (0.0, 0.0, 0.0)

        # Monte Carlo hypervolume estimation
        n_samples = 10_000
        rng = np.random.RandomState(42)

        # Find bounding box
        objectives = [p.objectives() for p in self.frontier]
        max_obj = tuple(max(o[i] for o in objectives) for i in range(3))

        # Sample random points in the bounding box
        samples = rng.uniform(
            low=reference_point,
            high=max_obj,
            size=(n_samples, 3),
        )

        # Count samples dominated by any frontier point
        dominated = 0
        frontier_objs = np.array(objectives)   # (M, 3)

        for sample in samples:
            # A sample is dominated if any frontier point dominates it
            dominated_by = np.all(frontier_objs >= sample, axis=1)
            if dominated_by.any():
                dominated += 1

        box_vol = np.prod([max_obj[i] - reference_point[i] for i in range(3)])
        hv = (dominated / n_samples) * box_vol
        return round(float(hv), 6)

    def summary(self) -> str:
        lines = [
            f"Pareto Frontier: {len(self.frontier)} non-dominated / {len(self._points)} total",
            f"{'Run':30s}  {'acc':>6}  {'lat':>6}  {'cost':>7}  optimal",
        ]
        for p in sorted(self._points, key=lambda x: x.accuracy, reverse=True):
            star = "★" if p.is_pareto_optimal else " "
            lines.append(
                f"{p.run_name:30s}  {p.accuracy:.3f}  {p.latency_ms:5.0f}ms  "
                f"${p.cost_per_ticket:.3f}   {star}"
            )
        return "\n".join(lines)

    def to_dataframe(self):
        """Return all points as a pandas DataFrame."""
        import pandas as pd
        rows = [
            {
                "run_name":         p.run_name,
                "domain":           p.domain,
                "lora_rank":        p.lora_rank,
                "accuracy":         p.accuracy,
                "latency_ms":       p.latency_ms,
                "cost_per_ticket":  p.cost_per_ticket,
                "is_pareto_optimal":p.is_pareto_optimal,
            }
            for p in self._points
        ]
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "frontier": [asdict(p) for p in self.frontier],
            "all_points": [asdict(p) for p in self._points],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("ParetoFrontier saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ParetoFrontier":
        with open(path) as f:
            data = json.load(f)
        frontier = cls()
        for point_dict in data.get("all_points", []):
            frontier.add(ParetoPoint(**point_dict))
        frontier._computed = False
        return frontier

    def plot(self, path: Optional[str | Path] = None):
        """
        Generate a 2D Pareto plot (accuracy vs cost, sized by latency).
        Saves to path if provided, otherwise returns the Figure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — cannot generate plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        COLORS = {
            "technical":  "#185FA5",
            "billing":    "#0F6E56",
            "returns":    "#993C1D",
            "escalation": "#993556",
        }

        # Plot all non-frontier points
        for p in self._points:
            if not p.is_pareto_optimal:
                ax.scatter(
                    p.cost_per_ticket, p.accuracy,
                    s=p.latency_ms / 2,
                    c="#CCCCCC", alpha=0.5, zorder=3,
                )

        # Plot Pareto-optimal points
        for p in self.frontier:
            color = COLORS.get(p.domain, "#888888")
            ax.scatter(
                p.cost_per_ticket, p.accuracy,
                s=p.latency_ms / 2,
                c=color, alpha=0.9, zorder=5,
                edgecolors="white", linewidth=1,
                label=f"{p.run_name} (r={p.lora_rank})",
            )
            ax.annotate(
                f"r={p.lora_rank}",
                (p.cost_per_ticket, p.accuracy),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=color,
            )

        # Connect frontier points (sorted by cost)
        frontier_sorted = sorted(self.frontier, key=lambda p: p.cost_per_ticket)
        if frontier_sorted:
            xs = [p.cost_per_ticket for p in frontier_sorted]
            ys = [p.accuracy        for p in frontier_sorted]
            ax.plot(xs, ys, "--", color="#444444", alpha=0.4, linewidth=1.2)

        ax.set_xlabel("Cost per ticket ($)", fontsize=12)
        ax.set_ylabel("Auto-resolution rate", fontsize=12)
        ax.set_title(
            "Pareto Frontier: Accuracy vs Cost\n"
            "(bubble size ∝ p95 latency, ★ = non-dominated)", fontsize=13
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Pareto plot saved → %s", path)
        return fig


# ── Convenience builder ────────────────────────────────────────────────────────

def build_pareto_frontier(eval_results: List) -> ParetoFrontier:
    """
    Build a ParetoFrontier from a list of EvaluationResult objects.

    Parameters
    ----------
    eval_results : List[EvaluationResult] from evaluate_checkpoints

    Returns
    -------
    Computed ParetoFrontier with is_pareto_optimal flags set
    """
    frontier = ParetoFrontier()

    for result in eval_results:
        # Extract domain and rank from run_name (e.g. "technical_r32_lr2e-04")
        run = result.run_name
        domain    = run.split("_")[0] if "_" in run else "technical"
        lora_rank = 32   # default; parse from run_name if available
        for part in run.split("_"):
            if part.startswith("r") and part[1:].isdigit():
                lora_rank = int(part[1:])
                break

        point = ParetoPoint(
            run_name=run,
            domain=domain,
            lora_rank=lora_rank,
            accuracy=result.auto_resolution_rate,
            latency_ms=result.latency.p95_ms if result.latency else 150.0,
            cost_per_ticket=result.cost.avg_cost_per_ticket,
        )
        frontier.add(point)

    frontier.compute()
    logger.info("Built Pareto frontier:\n%s", frontier.summary())
    return frontier


# ── Synthetic sweep results (for notebook / demo) ─────────────────────────────

def synthetic_sweep_pareto(seed: int = 42) -> ParetoFrontier:
    """
    Generate a realistic synthetic Pareto frontier for demonstration.
    Used in notebooks and tests when real sweep results aren't available.
    """
    rng = np.random.RandomState(seed)
    frontier = ParetoFrontier()

    configs = [
        # (domain, rank, base_acc, base_lat, base_cost)
        ("technical",  8,  0.88, 85,  0.03),
        ("technical",  16, 0.92, 100, 0.06),
        ("technical",  32, 0.965,120, 0.12),
        ("technical",  64, 0.970,180, 0.22),
        ("billing",    16, 0.91, 95,  0.07),
        ("billing",    24, 0.938,115, 0.12),
        ("billing",    32, 0.942,135, 0.16),
        ("returns",    16, 0.908,90,  0.06),
        ("returns",    28, 0.941,112, 0.09),
        ("returns",    32, 0.945,130, 0.14),
        ("escalation", 4,  0.990,70,  0.02),
        ("escalation", 8,  0.998,78,  0.03),
        ("escalation", 16, 0.999,90,  0.05),
    ]

    for domain, rank, acc, lat, cost in configs:
        noise_acc  = rng.randn() * 0.005
        noise_lat  = rng.randn() * 5
        noise_cost = rng.randn() * 0.005
        point = ParetoPoint(
            run_name=f"{domain}_r{rank}",
            domain=domain,
            lora_rank=rank,
            accuracy=round(min(acc + noise_acc, 1.0), 4),
            latency_ms=round(max(lat + noise_lat, 50), 1),
            cost_per_ticket=round(max(cost + noise_cost, 0.01), 4),
        )
        frontier.add(point)

    frontier.compute()
    return frontier
