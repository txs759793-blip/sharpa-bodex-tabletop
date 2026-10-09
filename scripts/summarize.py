"""Print a comparison table for runs under work_dir: python scripts/summarize.py baseline best"""
from pathlib import Path
import json, sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_paths

work = Path(load_paths()["work_dir"])
print("%-22s %7s %10s %12s %12s %14s %16s" % ("run", "scenes", "candidates", "official_kp", "a1_kp5", "ho_pene_p90mm", "self_pene>1mm"))
for name in sys.argv[1:]:
    rows = []
    for f in sorted((work / name / "chunks").glob("c[0-9]*/DONE.json")):
        rows += json.loads(f.read_text())
    if not rows:
        print(name, "no finished chunks"); continue
    s = np.array([r["succ"] for r in rows]); ho = np.array([r.get("ho", np.nan) for r in rows]) * 1000; hh = np.array([r.get("hh", np.nan) for r in rows]) * 1000
    k = work / name / "reeval_a1_kp5/summary.json"
    kp5 = ("%.2f%%" % json.loads(k.read_text())["success_percent"]) if k.exists() else "not run"
    print("%-22s %7d %10d %11.2f%% %12s %14.2f %15.1f%%" % (name, len({r["scene"] for r in rows}), len(rows), 100 * s.mean(), kp5,
                                                         np.nanpercentile(ho, 90), 100 * np.nanmean(hh > 1)))
