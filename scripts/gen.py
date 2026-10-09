"""Run BODex on a fixed list of scenes: python gen.py <chunk_dir> <bodex_root>  (chunk_dir holds rows.json and task.yml)."""
from pathlib import Path
import json, os, runpy, sys
from curobo.util import world_cfg_generator as worlds

run, repo = Path(sys.argv[1]), Path(sys.argv[2])
rows = json.loads((run / "rows.json").read_text())


def fixed_scenes(self, type, template_path, start, end):
    assert type == "scene_cfg"
    self.scene_path_lst = [r["scene"] for r in rows]


worlds.WorldConfigDataset.__init__ = fixed_scenes
sys.argv = ["plan_batch_env.py", "-c", str(run / "task.yml"), "-f", str(run), "-w", os.environ.get("PARALLEL_WORLD", "8"), "-m", "npy"]
runpy.run_path(str(repo / "example_grasp/plan_batch_env.py"), run_name="__main__")
