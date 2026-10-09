"""Shared helpers: load paths.yaml and write resolved configs (absolute paths) into <work_dir>/resolved."""
from pathlib import Path
import json, re, shutil
import yaml

REPO = Path(__file__).resolve().parent.parent


def load_paths(p=None):
    cfg = yaml.safe_load(Path(p or REPO / "paths.yaml").read_text())
    for k in ("bodex_root", "dexgraspbench_root", "sharpa_models_root", "python", "work_dir"):
        assert k in cfg, "missing %s in paths.yaml" % k
    return {k: str(v) for k, v in cfg.items()}


def hand_xml(paths):
    return Path(paths["sharpa_models_root"]) / "wave_01/right_sharpa_wave/right_sharpa_wave.xml"


def prepare(paths):
    """Write resolved configs. Returns the resolved directory. Safe to call repeatedly."""
    res = Path(paths["work_dir"]) / "resolved"
    res.mkdir(parents=True, exist_ok=True)

    def sub(text):
        return text.replace("__RESOLVED__", str(res)).replace("__SHARPA_MODELS__", paths["sharpa_models_root"])

    for src, dst in (("configs/robot/robot.yml", "robot.yml"), ("configs/robot/right_sharpa_wave_bodex.urdf", "right_sharpa_wave_bodex.urdf"),
                     ("configs/robot/spheres.yml", "spheres.yml"), ("configs/robot/hand_pose_transfer.yml", "hand_pose_transfer.yml"),
                     ("configs/generation/tabletop_baseline.yml", "tabletop_baseline.yml"), ("configs/generation/tabletop_best.yml", "tabletop_best.yml")):
        (res / dst).write_text(sub((REPO / src).read_text()))
    # optimizer settings of the best config: three edits on top of BODex's own gradient_grasp_fc.yml
    g = (Path(paths["bodex_root"]) / "src/curobo/content/configs/task/gradient_grasp_fc.yml").read_text()
    g, n1 = re.subn(r"(?m)^(  n_iters:)\s*\d+", r"\g<1> 1500", g)
    g, n2 = re.subn(r"(?m)^(  base_scale:)\s*\[0\.01, 0\.1, 0\.1\]", r"\g<1> [0.02, 0.2, 0.2]", g)
    assert n1 == 1 and n2 == 1, "unexpected gradient_grasp_fc.yml in BODex checkout"
    (res / "gradient_best.yml").write_text(g)
    # hand config for DexGraspBench (hydra reads it from its own config/hand folder)
    hand = Path(paths["dexgraspbench_root"]) / "config/hand/sharpa_wave.yaml"
    hand.write_text(sub((REPO / "configs/dexgraspbench/sharpa_wave.yaml").read_text()))
    assert hand_xml(paths).is_file(), "Sharpa MuJoCo XML not found: %s" % hand_xml(paths)
    return res


def kp_xml(paths, kp):
    """Copy of the official hand XML with every actuator kp set to one value (the official file is not modified)."""
    src = hand_xml(paths)
    out = Path(paths["work_dir"]) / "resolved" / ("right_sharpa_wave_kp%s.xml" % kp)
    if not out.exists():
        s = src.read_text()
        s, n1 = re.subn(r'meshdir="meshes/"', 'meshdir="%s/"' % (src.parent / "meshes"), s)
        s, n2 = re.subn(r'(<position [^>]*?kp=")[0-9.]+(")', r"\g<1>%s\g<2>" % kp, s)
        assert n1 == 1 and n2 == 22, (n1, n2)
        out.write_text(s)
    return out


def load_scenes(paths, name="scenes_tabletop_1000.json", start=0, end=None):
    rows = json.loads((REPO / "scenes" / name).read_text())[start:end]
    base = Path(paths["bodex_root"]) / "src/curobo/content/assets/object/DGN_2k/scene_cfg"
    for r in rows:
        r["scene"] = str(base / (r["scene_key"] + ".npy"))
    return rows
