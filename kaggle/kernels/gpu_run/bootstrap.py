# VPR-Space GPU session bootstrap (Internet OFF). Inputs (kernel_sources): vprspace-data-a, vprspace-data-b,
# vprspace-deps-core [, vprspace-deps-extra, vprspace-deps-anyloc]. The code of vprspace_repro/ is embedded below.
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile, time

SPEC = json.loads(r'''__SPEC__''')
CODE_B64 = "__CODE_B64__"
W = os.environ.get("VPRS_W", "/kaggle/working")  # overridable for local smoke tests
T0 = time.time()


def log(*a):
    print(f"{time.strftime('%H:%M:%S')} [+{time.time() - T0:6.0f}s]", *a, flush=True)


def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def main():
    os.makedirs(f"{W}/logs", exist_ok=True)
    code = "/tmp/vprcode"
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64))).extractall(code)
    log("spec", json.dumps(SPEC))
    parts = sorted(glob.glob("/kaggle/input/**/vpr-space.tar.0*", recursive=True), key=os.path.basename)
    names = [os.path.basename(p) for p in parts]
    assert names == [f"vpr-space.tar.{i:04d}" for i in range(11)], names
    # 1) extract the dataset to local disk (background) ...
    os.makedirs("/tmp/vprdata", exist_ok=True)
    ext = subprocess.Popen("cat " + " ".join(f"'{p}'" for p in parts) + " | tar -x -C /tmp/vprdata", shell=True)
    # 2) ... while preparing caches and the pinned environment
    tree = subprocess.run("find /kaggle/input -maxdepth 5 -type d | head -80", shell=True, capture_output=True, text=True).stdout
    log("input tree:\n" + tree)
    def find_dirs(name):  # kernel_sources are mounted at varying depths under /kaggle/input
        return sorted(d for d in glob.glob(f"/kaggle/input/**/{name}", recursive=True)
                      if os.path.isdir(d) and "/torch_home/" not in d and "/hf_home/" not in d and "/wheels/" not in d)
    for kind in ("torch_home", "hf_home"):
        os.makedirs(f"/tmp/{kind}", exist_ok=True)
        for src in find_dirs(kind):
            sh(["cp", "-rs", "--remove-destination", src + "/.", f"/tmp/{kind}/"])
    extra = find_dirs("extra")
    vme_dir = next((f"{e}/VPR-methods-evaluation" for e in extra if os.path.isdir(f"{e}/VPR-methods-evaluation")), None)
    mixvpr_root = "/tmp/mixvpr_root"
    mx = next((f"{e}/mixvpr" for e in extra if os.path.isdir(f"{e}/mixvpr")), None)
    if mx:
        os.makedirs(f"{mixvpr_root}/trained_models", exist_ok=True)
        if not os.path.exists(f"{mixvpr_root}/trained_models/mixvpr"):
            os.symlink(mx, f"{mixvpr_root}/trained_models/mixvpr")
    py = sys.executable
    if SPEC.get("env", "pinned") == "pinned":
        wheels = next(d for d in find_dirs("wheels") if glob.glob(f"{d}/torch-2.6.0*.whl"))
        req = sorted(glob.glob("/kaggle/input/**/requirements_pinned.txt", recursive=True))[0]
        log("wheels:", wheels, "req:", req)
        try:
            sh([sys.executable, "-m", "venv", "/tmp/vprenv"])
            pip = ["/tmp/vprenv/bin/python", "-m", "pip"]
        except subprocess.CalledProcessError:
            shutil.rmtree("/tmp/vprenv", ignore_errors=True)
            sh([sys.executable, "-m", "venv", "--without-pip", "/tmp/vprenv"])
            pip = [sys.executable, "-m", "pip", "--python", "/tmp/vprenv/bin/python"]
        sh(pip + ["install", "-q", "--no-index", "--find-links", wheels, "-r", req])
        sh(pip + ["install", "-q", "--no-index", "--no-deps", "--find-links", wheels, "pytorch-lightning", "lightning-utilities",
                  "torchmetrics", "fast-pytorch-kmeans", "timm", "psutil", "pytorch-metric-learning"])
        py = "/tmp/vprenv/bin/python"
    # the venv is created --without-pip on Kaggle, so "venv/bin/python -m pip" is empty: use the pip that installed it
    freeze_cmd = (pip + ["freeze"]) if py != sys.executable else [sys.executable, "-m", "pip", "freeze"]
    freeze = subprocess.run(freeze_cmd, capture_output=True, text=True).stdout
    open(f"{W}/logs/pip_freeze.txt", "w").write(freeze)
    rc = ext.wait()
    assert rc == 0, f"tar extraction failed ({rc})"
    root = os.path.dirname(glob.glob("/tmp/vprdata/**/vpr_gt", recursive=True)[0])
    log("data root", root, "extracted")
    # 3) load-balanced assignment of trajectories to GPUs (LPT on #images)
    n_gpu = int(subprocess.run("nvidia-smi -L 2>/dev/null | grep -c GPU", shell=True, capture_output=True, text=True).stdout.strip() or 0)
    has_gpu = n_gpu > 0
    n_gpu = max(n_gpu, 1)  # CPU fallback: one worker
    trajs = sorted(os.listdir(f"{root}/images"))
    if SPEC.get("trajectories"):
        trajs = [t for t in trajs if t in SPEC["trajectories"]]
    def png_wh(path):
        with open(path, "rb") as f:
            head = f.read(24)
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    native = any(not isinstance(SPEC.get("sizes", {}).get(c, [0]), list) for c in SPEC["configs"]) or SPEC.get("pixel_cost", False)
    cost = {}
    for t in trajs:
        files = sorted(glob.glob(f"{root}/images/{t}/*.png"))
        w, h = png_wh(files[0])
        cost[t] = len(files) * ((w * h / 1e6) if native else 1.0)
    assign, load = [[] for _ in range(n_gpu)], [0] * n_gpu
    for t in sorted(trajs, key=lambda t: -cost[t]):
        k = load.index(min(load))
        assign[k].append(t)
        load[k] += cost[t]
    log(f"{n_gpu} GPUs, images per GPU {load}")
    desc = f"{W}/desc"
    os.makedirs(desc, exist_ok=True)
    spec = {"data_root": root, "desc_dir": desc, "configs": SPEC["configs"], "assign": assign,
            "vme_dir": vme_dir, "mixvpr_root": mixvpr_root}
    json.dump(spec, open("/tmp/worker_spec.json", "w"))
    cpus = len(os.sched_getaffinity(0))
    hf_plain = ":".join(find_dirs("hf_plain"))
    log("hf_plain dirs:", hf_plain or "none")
    env = dict(os.environ, TORCH_HOME="/tmp/torch_home", HF_HOME="/tmp/hf_home", HF_HUB_OFFLINE="1", VPRS_HF_PLAIN=hf_plain,
               TRANSFORMERS_OFFLINE="1", PYTHONPATH=f"{code}:{code}/upstream", VPRS_SESSION=SPEC["session"],
               VPRS_LOADER_WORKERS=str(max(2, (cpus - 2) // max(n_gpu, 1))), OMP_NUM_THREADS="4")
    def run_steps(key):
        for step in SPEC.get(key, []):
            argv = [py] + [x.format(root=root, desc=desc, code=code) for x in step]
            e = dict(env, CUDA_VISIBLE_DEVICES="0") if has_gpu else dict(env)
            log(f"{key} step:", " ".join(argv))
            subprocess.run(argv, env=e, check=False)
    run_steps("pre")
    procs = []
    for i in range(n_gpu):
        if not assign[i]:
            continue
        e = dict(env, CUDA_VISIBLE_DEVICES=str(i)) if has_gpu else dict(env)
        procs.append(subprocess.Popen([py, f"{code}/worker.py", "--spec", "/tmp/worker_spec.json", "--worker", str(i)],
                                      env=e, stdout=open(f"{W}/logs/worker{i}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    log("workers exit codes", rcs)
    for i in range(n_gpu):
        if os.path.exists(f"{W}/logs/worker{i}.log"):
            print(open(f"{W}/logs/worker{i}.log").read()[-6000:], flush=True)
    run_steps("post")
    # 4) metrics with the authors' evaluation code
    sh([py, f"{code}/metrics_all.py", "--root", root, "--desc-dir", desc, "--out", f"{W}/results", "--tag", SPEC["session"]], env=env)
    info = {"session": SPEC, "seconds": round(time.time() - T0, 1), "n_gpu": n_gpu, "cpus": cpus, "assign": assign,
            "images_per_gpu": load, "worker_exit": rcs,
            "nvidia_smi": subprocess.run("nvidia-smi || true", shell=True, capture_output=True, text=True).stdout}
    json.dump(info, open(f"{W}/results/_session_{SPEC['session']}.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
