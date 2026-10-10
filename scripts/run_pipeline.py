"""Generate grasps with BODex, convert them, and evaluate them with DexGraspBench (tabletop, default criteria).

Example:
  python scripts/run_pipeline.py --config tabletop_best --name best --start 0 --end 1000 --gpu 0

Outputs go to <work_dir>/<name>/ . Re-running resumes from finished chunks.
A chunk is one BODex process (default 40 scenes, solved 8 scenes at a time).
"""
from pathlib import Path
from datetime import datetime
import argparse, json, os, subprocess, sys, time
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_paths, load_scenes, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="tabletop_best", help="tabletop_best or tabletop_baseline")
ap.add_argument("--name", required=True, help="output folder name under work_dir")
ap.add_argument("--scenes", default="scenes_tabletop_1000.json")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=1000)
ap.add_argument("--gpu", default="0")
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--paths", default=None)
args = ap.parse_args()

paths = load_paths(args.paths)
PY, BODEX, BENCH = paths["python"], Path(paths["bodex_root"]), Path(paths["dexgraspbench_root"])
RUN = Path(paths["work_dir"]) / args.name
T0 = time.time()


def status(**kw):
    d = dict(time=datetime.now().isoformat(timespec="seconds"), elapsed_min=round((time.time() - T0) / 60, 1), **kw)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "status.json").write_text(json.dumps(d, indent=1))
    print(json.dumps(d), flush=True)


def convert_scene(raw_path, row, joint_order, grasp_root, n_expected):
    """BODex raw output (one file per scene) -> one DexGraspBench grasp file per candidate."""
    from task.convert_format import load_scene_cfg
    raw = np.load(raw_path, allow_pickle=True).item()
    pose = np.asarray(raw["robot_pose"])
    if pose.shape != (1, n_expected, 3, 29):
        raise RuntimeError("unexpected raw pose shape in %s" % raw_path)
    # BODex occasionally returns non-finite poses for a few candidates (its log then shows a GJK warning).
    # Such candidates are not written, so they get no evaluation file and are counted as failures.
    valid = np.isfinite(pose[0]).all(axis=(1, 2))
    saved = list(raw["joint_names"])
    order = [saved.index(n) for n in joint_order]
    pose = np.concatenate([pose[0, ..., :7], pose[0, ..., 7:][..., order]], axis=-1)
    scene = load_scene_cfg(row["scene"])
    assert scene["scene_id"] == row["scene_key"], row["scene_key"]
    obj = scene["scene"][scene["task"]["obj_name"]]
    common = dict(obj_path=str(Path(obj["file_path"]).resolve().parent.parent), obj_scale=float(np.asarray(obj["scale"], float)[0]),
                  obj_pose=obj["pose"], scene_path=str(Path(row["scene"]).resolve()))
    for i, traj in enumerate(pose):
        if not valid[i]:
            continue
        out = grasp_root / row["scene_key"] / ("%d_grasp.npy" % i)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, dict(common, pregrasp_qpos=traj[0], grasp_qpos=traj[1], squeeze_qpos=traj[2]))


def main():
    res = prepare(paths)
    os.chdir(BENCH)
    sys.path.insert(0, str(BENCH / "src"))
    import mujoco
    hand = yaml.safe_load((BENCH / "config/hand/sharpa_wave.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(hand["xml_path"])
    joint_order = [model.joint(i).name for i in range(model.njnt)]
    rows = load_scenes(paths, args.scenes, args.start, args.end)
    task = yaml.safe_load((res / (args.config + ".yml")).read_text())
    nseed = int(task["seed_num"])
    allrows = []
    for ci, start in enumerate(range(0, len(rows), args.chunk)):
        batch = rows[start:start + args.chunk]
        cdir = RUN / "chunks" / ("c%03d" % ci)
        done = cdir / "DONE.json"
        if not done.exists():
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "rows.json").write_text(json.dumps(batch))
            (cdir / "task.yml").write_text(yaml.safe_dump(dict(task, world=dict(task["world"], start=0, end=len(batch))), sort_keys=False))
            expected = [cdir / "grasp_data" / (r["scene_key"] + "_grasp.npy") for r in batch]
            for attempt in range(3):
                if all(p.is_file() for p in expected):
                    break
                status(stage="GENERATING", chunk=ci, attempt=attempt, scenes_done=start, scenes_total=len(rows))
                with open(cdir / "generation.log", "a") as log:
                    subprocess.call([PY, "-u", str(REPO / "scripts/gen.py"), str(cdir), str(BODEX)], cwd=BODEX,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu), stdout=log, stderr=subprocess.STDOUT)
            if not all(p.is_file() for p in expected):
                raise RuntimeError("generation incomplete, see %s" % (cdir / "generation.log"))
            save = cdir / "results"
            grasp_root = save / "full_sharpa_wave/graspdata"
            for r, raw in zip(batch, expected):
                convert_scene(raw, r, joint_order, grasp_root, nseed)
            status(stage="EVALUATING", chunk=ci, scenes_done=start, scenes_total=len(rows))
            with open(cdir / "evaluation.log", "a") as log:
                rc = subprocess.call([PY, "-u", str(BENCH / "src/main.py"), "hand=sharpa_wave", "setting=tabletop", "exp_name=full",
                                      "save_root=" + str(save), "n_worker=%d" % args.workers, "skip=true", "task=eval", "task.max_num=-1",
                                      "task.debug_render=false", "task.debug_viewer=false"],
                                     cwd=BENCH, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=log, stderr=subprocess.STDOUT)
            if rc != 0:
                raise RuntimeError("evaluation failed, see %s" % (cdir / "evaluation.log"))
            out = []
            for r, rawp in zip(batch, expected):
                ge = np.asarray(np.load(rawp, allow_pickle=True).item()["grasp_error"])[0].max(-1)
                for i in range(nseed):
                    e = save / "full_sharpa_wave/evaluation" / r["scene_key"] / ("%d_grasp.npy" % i)
                    rec = dict(scene=r["scene_key"], i=i, ge=float(ge[i]))
                    if e.is_file():
                        d = np.load(e, allow_pickle=True).item()
                        rec.update(succ=bool(d["succ_flag"]), dp=float(d["delta_pos"]), da=float(d["delta_angle"]), ho=float(d["ho_pene"]),
                                   hh=float(d["self_pene"]), cn=int(d["contact_num"]))
                    else:  # the evaluator raised for this candidate: counted as a failure
                        rec.update(evaluated=False, succ=False)
                    out.append(rec)
            done.write_text(json.dumps(out))
        allrows += json.loads(done.read_text())
        s = np.array([r["succ"] for r in allrows])
        status(stage="CHUNK_DONE", chunk=ci, scenes_done=min(start + args.chunk, len(rows)), scenes_total=len(rows),
               candidates=len(s), success=int(s.sum()), success_percent_official_kp=round(100 * s.mean(), 2))
    s = np.array([r["succ"] for r in allrows])
    status(stage="FINISHED", scenes_done=len(rows), scenes_total=len(rows), candidates=len(s), success=int(s.sum()),
           success_percent_official_kp=round(100 * s.mean(), 2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        status(stage="STOPPED_ERROR", error=repr(e))
        raise
