"""Re-evaluate already generated grasps with a squeeze scale (alpha) and a uniform hand kp. Generation is untouched.

Example (the standard condition used in the report):
  python scripts/reeval_alpha_kp.py --name best --alpha 1 --kp 5

alpha: squeeze joints = grasp + alpha * (grasp - pregrasp); alpha=1 is what BODex saves.
kp:    a number sets every actuator kp of the hand to that value (a copy of the official XML is used);
       'official' keeps the per-joint values of the Sharpa XML.
Prints the success rate over ALL candidates (candidates whose evaluation raised are counted as failures).
"""
from pathlib import Path
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import kp_xml, load_paths

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--kp", default="5")
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--paths", default=None)
args = ap.parse_args()
paths = load_paths(args.paths)
BENCH = Path(paths["dexgraspbench_root"])
run = Path(paths["work_dir"]) / args.name
extra = [] if args.kp == "official" else ["hand.xml_path=" + str(kp_xml(paths, args.kp))]
out = run / ("reeval_a%g_kp%s" % (args.alpha, args.kp))
tot = succ = 0
for c in sorted((run / "chunks").glob("c[0-9]*")):
    if not (c / "DONE.json").exists():
        continue
    grasp = c / "results/full_sharpa_wave/graspdata"
    save = out / c.name
    if args.alpha != 1.0:
        g2 = save / "graspdata_alpha"
        for p in grasp.rglob("*.npy"):
            q = g2 / p.relative_to(grasp)
            if q.exists():
                continue
            d = np.load(p, allow_pickle=True).item()
            sq = np.array(d["grasp_qpos"], dtype=float)
            sq[7:] += args.alpha * (np.asarray(d["grasp_qpos"])[7:] - np.asarray(d["pregrasp_qpos"])[7:])
            d["squeeze_qpos"] = sq.astype(np.asarray(d["squeeze_qpos"]).dtype)
            q.parent.mkdir(parents=True, exist_ok=True)
            np.save(q, d)
        grasp = g2
    save.mkdir(parents=True, exist_ok=True)
    with open(save / "evaluation.log", "a") as log:
        rc = subprocess.call([paths["python"], "-u", str(BENCH / "src/main.py"), "hand=sharpa_wave", "setting=tabletop", "exp_name=full",
                              "save_root=" + str(save), "grasp_dir=" + str(grasp), "n_worker=%d" % args.workers, "skip=true", "task=eval",
                              "task.max_num=-1", "task.debug_render=false", "task.debug_viewer=false"] + extra,
                             cwd=BENCH, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=log, stderr=subprocess.STDOUT)
    assert rc == 0, save
    tot += len(json.loads((c / "DONE.json").read_text()))  # all generated candidates, including ones that could not be evaluated
    succ += sum(bool(np.load(p, allow_pickle=True).item()["succ_flag"]) for p in (save / "full_sharpa_wave/evaluation").rglob("*.npy"))
res = dict(name=args.name, setting="tabletop", alpha=args.alpha, kp=args.kp, candidates=tot, success=succ, success_percent=round(100 * succ / max(tot, 1), 2))
(out / "summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
