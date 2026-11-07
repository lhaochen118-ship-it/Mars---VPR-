"""Push the given plan sessions, retrying every 30 s while the account's GPU slots are busy
("Maximum batch GPU session count") or its vprspace-deps-core is not COMPLETE yet.

python3 kaggle/push_pending.py kaggle/wave2_plan.json <session> [<session> ...]
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from launch import USERNAMES, metadata  # noqa: E402
import launch_session  # noqa: E402

SPLIT = "/tmp/claude-0/-home-user-Mars---VPR-/f4f85d11-307a-560d-8600-f8fcabc2f136/scratchpad/anyloc_demo_split.json"


def try_push(s, split):
    import shutil, tempfile
    sys.path.insert(0, os.path.join(launch_session.REPO, "vprspace_repro"))
    from configs import CONFIGS
    trajs = s.get("trajectories") or (split[s["split"]] if "split" in s else None)
    pixel_cost = any(isinstance(CONFIGS[c].get("size"), str) and not CONFIGS[c]["size"].startswith("pilbilinear") for c in s["configs"])
    spec = {"session": s["session"], "env": s["env"], "configs": s["configs"], "trajectories": trajs, "pixel_cost": pixel_cost}
    if any(CONFIGS[c]["kind"] == "densevlad" for c in s["configs"]):
        spec["pre"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "vocab"]]
        spec["post"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "pca"]]
    for key in ("pre", "post"):  # custom steps given by the plan (e.g. the CPU-parallel VLFeat DenseVLAD)
        if key in s:
            spec[key] = s[key]
    user = USERNAMES[s["label"]]
    sources = [f"{user}/vprspace-data-a", f"{user}/vprspace-data-b"] + [f"{user}/vprspace-deps-{d}" for d in s["deps"]]
    sources += [f"{user}/{k}" for k in s.get("extra_sources", [])]
    d = tempfile.mkdtemp()
    open(os.path.join(d, "run.py"), "w").write(launch_session.build_script(spec))
    json.dump(metadata(s["label"], f"vprspace-run-{s['session']}", "run.py", gpu=s["gpu"], kernel_sources=sources,
                       dataset_sources=s.get("dataset_sources", [])),
              open(os.path.join(d, "kernel-metadata.json"), "w"))
    if os.environ.get("VPRS_DRY_RUN"):
        md = json.load(open(os.path.join(d, "kernel-metadata.json")))
        print(json.dumps({"id": md["id"], "machine_shape": md.get("machine_shape"), "internet": md["enable_internet"],
                          "competition": md["competition_sources"], "kernel_sources": md["kernel_sources"],
                          "dataset_sources": md["dataset_sources"],
                          "spec": {k: v for k, v in spec.items() if k != "configs"}, "n_configs": len(spec["configs"])}))
        shutil.rmtree(d, ignore_errors=True)
        return "successfully pushed (dry-run)"
    r = subprocess.run(["kg", s["label"], "kernels", "push", "-p", d], capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    return (r.stdout + r.stderr).strip().splitlines()[-1]


def main():
    plan = {s["session"]: s for s in json.load(open(sys.argv[1]))["sessions"]}
    todo = list(sys.argv[2:])
    split = json.load(open(SPLIT))
    while todo:
        for name in list(todo):
            s = plan[name]
            st = subprocess.run(["kg", s["label"], "kernels", "status", f"{USERNAMES[s['label']]}/vprspace-deps-core"], capture_output=True, text=True).stdout
            if "COMPLETE" not in st:
                continue
            msg = try_push(s, split)
            if "successfully pushed" in msg:
                print(f"{time.strftime('%H:%M:%S')} LAUNCHED {name}: {msg}", flush=True)
                todo.remove(name)
            elif "Maximum batch GPU session" not in msg:
                print(f"{time.strftime('%H:%M:%S')} PUSH ERROR {name}: {msg}", flush=True)
                todo.remove(name)
        if todo:
            time.sleep(30)
    print("ALL PUSHED", flush=True)


if __name__ == "__main__":
    main()
