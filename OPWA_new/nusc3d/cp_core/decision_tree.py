"""
Decision Tree — collapse判定阈值逻辑.

Ported from OPWA_v3/exp0/decision_tree.py.
Evaluates CP results against a decision threshold hierarchy
to determine whether a coverage-collapse phenomenon is significant.

Thresholds:
    bin0_ratio >= 0.05  :  the hardest bin has enough points to matter
    bin0_gap  >= 0.20   :  phenomenon is significant
    bin0_gap  >= 0.10   :  phenomenon exists but is weak
    bin0_gap  <  0.10   :  not significant, stop
"""

from typing import Optional


class DecisionTree:
    """Evaluate results against the coverage-collapse decision tree.

    Usage:
        tree = DecisionTree(results_dict)
        tree.evaluate()
        print(tree.summary)
        # tree.decision is one of:
        #   "significant", "weak", "negative", "insufficient_bin0"
    """

    BIN0_RATIO_THRESHOLD = 0.05
    GAP_SIGNIFICANT = 0.20
    GAP_WEAK = 0.10

    def __init__(self, results: dict):
        self.results = results
        self.decision: Optional[str] = None
        self.summary: Optional[str] = None

    def evaluate(self) -> str:
        """Run the decision tree and populate summary / decision."""
        d = self.results
        lines = []

        lines.append("=" * 60)
        lines.append("DECISION TREE EVALUATION — COVERAGE COLLAPSE")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  bin0_ratio = {d['bin0_ratio']:.4f}  (threshold >= {self.BIN0_RATIO_THRESHOLD})")
        lines.append(f"  bin0_gap   = {d['bin0_gap']:.4f}")
        lines.append("")

        # Level 1: bin0 ratio check
        lines.append(f"  ┌─ bin0_ratio >= {self.BIN0_RATIO_THRESHOLD} ?")
        if d["bin0_ratio"] >= self.BIN0_RATIO_THRESHOLD:
            lines.append(f"  │  └─ YES ({d['bin0_ratio']:.4f} >= {self.BIN0_RATIO_THRESHOLD})")
            lines.append(f"  │")

            # Level 2: gap check
            lines.append(f"  │  ┌─ bin0_gap >= {self.GAP_SIGNIFICANT} ?")
            if d["bin0_gap"] >= self.GAP_SIGNIFICANT:
                lines.append(f"  │  │  └─ ✦ YES (gap={d['bin0_gap']:.4f})")
                lines.append(f"  │  │")
                lines.append(f"  │  │  >>> PHENOMENON SIGNIFICANT!")
                lines.append(f"  │  │  >>> Apply corrective method (PS-CRC, etc.)")
                self.decision = "significant"
            else:
                lines.append(f"  │  │  └─ NO (gap={d['bin0_gap']:.4f} < {self.GAP_SIGNIFICANT})")
                lines.append(f"  │  │")
                lines.append(f"  │  │  ┌─ bin0_gap >= {self.GAP_WEAK} ?")
                if d["bin0_gap"] >= self.GAP_WEAK:
                    lines.append(f"  │  │  │  └─ YES (gap={d['bin0_gap']:.4f} >= {self.GAP_WEAK})")
                    lines.append(f"  │  │  │")
                    lines.append(f"  │  │  │  >>> PHENOMENON EXISTS BUT WEAK")
                    lines.append(f"  │  │  │  >>> Consider harder OOD data")
                    lines.append(f"  │  │  │  >>> Wait for human decision")
                    self.decision = "weak"
                else:
                    lines.append(f"  │  │  │  └─ NO (gap={d['bin0_gap']:.4f} < {self.GAP_WEAK})")
                    lines.append(f"  │  │  │")
                    lines.append(f"  │  │  │  >>> PHENOMENON NOT SIGNIFICANT")
                    lines.append(f"  │  │  │  >>> Stop. Wait for human direction.")
                    self.decision = "negative"
        else:
            lines.append(f"  │  └─ NO (bin0_ratio={d['bin0_ratio']:.4f} < {self.BIN0_RATIO_THRESHOLD})")
            lines.append(f"  │")
            lines.append(f"  │  >>> Bin 0 has too few points. Binning may be wrong.")
            lines.append(f"  │  >>> Check difficulty scoring / bin edges.")
            self.decision = "insufficient_bin0"

        lines.append("")
        lines.append(f"  Decision: {self.decision}")
        lines.append("=" * 60)

        self.summary = "\n".join(lines)
        return self.summary

    def print_summary(self) -> None:
        if self.summary is None:
            self.evaluate()
        print(self.summary)
