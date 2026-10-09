"""Interactive 3D viewer (Viser) to compare two runs side by side on the same tabletop scenes.

  pip install viser trimesh mujoco numpy pyyaml      (a separate virtual env is recommended)
  python scripts/viewer.py --left baseline --right best [--port 8768]
Then forward the port (ssh -L 8768:127.0.0.1:8768 <server>) and open http://127.0.0.1:8768
Read-only: it only loads existing evaluation files written by run_pipeline.py.
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import mujoco, trimesh, viser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, hand_xml, load_paths

ap = argparse.ArgumentParser()
ap.add_argument("--left", required=True); ap.add_argument("--right", required=True)
ap.add_argument("--port", type=int, default=8768); ap.add_argument("--paths", default=None)
args = ap.parse_args()
paths = load_paths(args.paths)
WORK = Path(paths["work_dir"])
XML = str(hand_xml(paths))
PORT = args.port
GROUPS = [(args.left, args.left, 0.0), (args.right, args.right, 0.55)]
STAGES = {"Pregrasp · 预抓取": "pregrasp_qpos", "Grasp · 抓取": "grasp_qpos", "Squeeze · 挤压": "squeeze_qpos"}

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
if not geoms:
    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]


def geom_mesh(g):
    m = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]].copy()
    f = model.mesh_face[model.mesh_faceadr[m]: model.mesh_faceadr[m] + model.mesh_facenum[m]].copy()
    return v, f


def hand_poses(qpos, offset):
    """World pose (wxyz, xyz) of every hand geom for a 29-dim qpos (root xyz, root wxyz, 22 joints)."""
    data.qpos[:] = qpos[7:]
    mujoco.mj_forward(model, data)
    rq = np.asarray(qpos[3:7], float) / np.linalg.norm(qpos[3:7])
    out = []
    for g in geoms:
        gq = np.zeros(4); mujoco.mju_mat2Quat(gq, data.geom_xmat[g])
        wq = np.zeros(4); mujoco.mju_mulQuat(wq, rq, gq)
        wp = np.zeros(3); mujoco.mju_rotVecQuat(wp, data.geom_xpos[g], rq)
        out.append((wq, wp + np.asarray(qpos[:3], float) + offset))
    return out


# index: scene -> evaluation folder, per group
index = []
for label, pat, dy in GROUPS:
    d = {}
    for run in [WORK / pat]:
        for ev in run.glob("chunks/c[0-9]*/results/full_sharpa_wave/evaluation/*/*/*"):
            d["/".join(ev.parts[-3:])] = ev
    index.append(d)
order = [r["scene_key"] for r in json.loads((REPO / "scenes/scenes_tabletop_1000.json").read_text())]
scenes = [k for k in order if all(k in d for d in index)]
print("scenes available in both groups:", len(scenes), flush=True)
cache = {}


def load(scene):
    if scene not in cache:
        if len(cache) > 40:
            cache.clear()
        cache[scene] = [{int(p.name.split("_")[0]): np.load(p, allow_pickle=True).item() for p in d[scene].glob("*_grasp.npy")} for d in index]
    return cache[scene]


server = viser.ViserServer(host="127.0.0.1", port=PORT)
server.scene.set_up_direction("+z")
server.gui.configure_theme(dark_mode=True)
server.gui.add_markdown("**Sharpa Wave · 桌面抓取**\n\n左：原始参数；右：方案 B（迭代 1500、步长加倍、第一阶段 80%）。同一个场景，各自的第 k 个候选。成功/失败来自 DexGraspBench 桌面评测（抬起 10 cm）。")
g_scene = server.gui.add_slider("场景编号", min=0, max=max(len(scenes) - 1, 1), step=1, initial_value=0)
g_prev = server.gui.add_button("上一个场景")
g_next = server.gui.add_button("下一个场景")
g_cand = server.gui.add_slider("候选编号（0 = 优化器认为最好）", min=0, max=31, step=1, initial_value=0)
g_stage = server.gui.add_dropdown("目标姿态", options=list(STAGES), initial_value="Grasp · 抓取")
g_obj = server.gui.add_checkbox("显示物体与桌面", initial_value=True)
g_info = server.gui.add_markdown("")

hand_handles = [[], []]
for gi, (label, pat, dy) in enumerate(GROUPS):
    for j, g in enumerate(geoms):
        v, f = geom_mesh(g)
        hand_handles[gi].append(server.scene.add_mesh_simple(f"/hand{gi}/g{j}", vertices=v, faces=f, color=(205, 208, 220)))
state = {"scene": None, "extra": []}


def redraw(_=None):
    if not scenes:
        return
    scene = scenes[int(g_scene.value)]
    both = load(scene)
    k = int(g_cand.value)
    lines = [f"**{int(g_scene.value) + 1} / {len(scenes)}** · 候选 {k}", f"`{scene}`", ""]
    if state["scene"] != (scene, g_obj.value):
        for h in state["extra"]:
            h.remove()
        state["extra"] = []
        e0 = both[0][0]
        if g_obj.value:
            mesh = trimesh.load(str(Path(e0["obj_path"]) / "mesh/simplified.obj"), force="mesh", process=False)
            v = np.asarray(mesh.vertices) * float(e0["obj_scale"])
            pose = np.asarray(e0["obj_pose"], float)
            for gi, (label, pat, dy) in enumerate(GROUPS):
                off = np.array([0.0, dy, 0.0])
                state["extra"].append(server.scene.add_mesh_simple(f"/obj{gi}", vertices=v, faces=np.asarray(mesh.faces), color=(232, 140, 60),
                                                                   wxyz=pose[3:7], position=pose[:3] + off))
                state["extra"].append(server.scene.add_box(f"/table{gi}", dimensions=(0.5, 0.5, 0.02), color=(70, 78, 100),
                                                           position=(pose[0], pose[1] + dy, -0.01)))
        state["scene"] = (scene, g_obj.value)
    for h in state.get("labels", []):
        h.remove()
    state["labels"] = []
    for gi, (label, pat, dy) in enumerate(GROUPS):
        cand = both[gi]
        e = cand[k]
        off = np.array([0.0, dy, 0.0])
        for h, (wq, wp) in zip(hand_handles[gi], hand_poses(np.asarray(e[STAGES[g_stage.value]], float), off)):
            h.wxyz, h.position = wq, wp
        ok = bool(e["succ_flag"])
        nok = sum(bool(c["succ_flag"]) for c in cand.values())
        pose = np.asarray(e["obj_pose"], float)
        state["labels"].append(server.scene.add_label(f"/label{gi}", f"{label} · {'✔ 成功' if ok else '✘ 失败'}", position=(pose[0] + 0.22, pose[1] + dy, 0.02)))
        lines.append(f"{label}：{'✔ 成功' if ok else '✘ 失败'}（本场景 {nok}/32 成功，接触部件数 {int(e['contact_num'])}，穿透 {float(e['ho_pene']) * 1000:.1f} mm）")
    g_info.content = "\n\n".join(lines)


for h in (g_scene, g_cand, g_stage, g_obj):
    h.on_update(redraw)
g_prev.on_click(lambda _: setattr(g_scene, "value", max(int(g_scene.value) - 1, 0)))
g_next.on_click(lambda _: setattr(g_scene, "value", min(int(g_scene.value) + 1, len(scenes) - 1)))


@server.on_client_connect
def _(client):
    if scenes:
        p = np.asarray(load(scenes[int(g_scene.value)])[0][0]["obj_pose"], float)
        client.camera.position = (p[0] + 0.55, p[1] - 0.25, 0.4)
        client.camera.look_at = (p[0], p[1] + 0.27, 0.05)


redraw()
print("viewer ready on port", PORT, flush=True)
while True:
    time.sleep(3600)
