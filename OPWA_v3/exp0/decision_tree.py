import json
import os

from config import OUTPUT_DIR, TARGET_COVERAGE


DECISION_HTML = """
<style>
  .tree { font-family: 'Courier New', monospace; line-height: 1.6; }
  .yes { color: green; font-weight: bold; }
  .no { color: red; font-weight: bold; }
  .stop { background: #fff3f3; padding: 4px 8px; border-left: 4px solid red; }
  .signif { background: #f0fff0; padding: 4px 8px; border-left: 4px solid green; }
  .weak { background: #fffff0; padding: 4px 8px; border-left: 4px solid orange; }
</style>
"""


class DecisionTree:
    """Evaluate exp0_results.json against the decision tree."""

    def __init__(self, results_path=None):
        if results_path is None:
            results_path = os.path.join(OUTPUT_DIR, "exp0_results.json")
        self.results_path = results_path
        self.data = None
        self.decision = None

    def load(self):
        with open(self.results_path) as f:
            self.data = json.load(f)
        return self.data

    def evaluate(self):
        """Run the decision tree and return a text summary."""
        if self.data is None:
            self.load()

        d = self.data
        lines = []

        lines.append("=" * 60)
        lines.append("DECISION TREE EVALUATION")
        lines.append("=" * 60)
        lines.append(f"")
        lines.append(f"  bin0_pixel_ratio = {d['bin0_pixel_ratio']:.4f}  (threshold >= 0.05)")
        lines.append(f"  bin0_gap         = {d['bin0_gap']:.4f}")
        lines.append(f"")

        # Level 1: pixel ratio check
        lines.append(f"  ┌─ bin0_pixel_ratio >= 0.05 ?")
        if d["bin0_pixel_ratio"] >= 0.05:
            lines.append(f"  │  └─ YES ({d['bin0_pixel_ratio']:.4f} >= 0.05)")
            lines.append(f"  │")
            # Level 2: gap check
            lines.append(f"  │  ┌─ bin0_gap >= 0.20 ?")
            if d["bin0_gap"] >= 0.20:
                lines.append(f"  │  │  └─ ✦ YES (gap={d['bin0_gap']:.4f})")
                lines.append(f"  │  │")
                lines.append(f"  │  │  >>> PHENOMENON SIGNIFICANT!")
                lines.append(f"  │  │  >>> Ready for Experiment 1 (PS-CRC)")
                self.decision = "significant"
            else:
                lines.append(f"  │  │  └─ NO (gap={d['bin0_gap']:.4f} < 0.20)")
                lines.append(f"  │  │")
                lines.append(f"  │  │  ┌─ bin0_gap >= 0.10 ?")
                if d["bin0_gap"] >= 0.10:
                    lines.append(f"  │  │  │  └─ YES (gap={d['bin0_gap']:.4f} >= 0.10)")
                    lines.append(f"  │  │  │")
                    lines.append(f"  │  │  │  >>> PHENOMENON EXISTS BUT WEAK")
                    lines.append(f"  │  │  │  >>> Consider ACDC Fog dataset")
                    lines.append(f"  │  │  │  >>> Wait for human decision")
                    self.decision = "weak"
                else:
                    lines.append(f"  │  │  │  └─ NO (gap={d['bin0_gap']:.4f} < 0.10)")
                    lines.append(f"  │  │  │")
                    lines.append(f"  │  │  │  >>> PHENOMENON NOT SIGNIFICANT (gap={d['bin0_gap']:.4f})")
                    lines.append(f"  │  │  │  >>> Stop. Wait for human direction.")
                    self.decision = "negative"
        else:
            lines.append(f"  │  └─ NO (pixel_ratio={d['bin0_pixel_ratio']:.4f} < 0.05)")
            lines.append(f"  │")
            lines.append(f"  │  >>> Bin 0 has too few pixels. Transmittance may be wrong.")
            lines.append(f"  │  >>> Check depth direction and BETA value.")
            self.decision = "insufficient_bin0"

        lines.append(f"")
        lines.append(f"  Decision: {self.decision}")
        lines.append(f"=" * 60)

        return "\n".join(lines)

    def print_summary(self):
        print(self.evaluate())
