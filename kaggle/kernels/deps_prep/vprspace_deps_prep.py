# VPR-Space offline dependencies (CPU + Internet). PROFILE selects what to fetch:
#   core   : pinned wheels (authors' requirements subset + extras) tested in a clean venv,
#            torch.hub caches for BoQ, MegaLoc (HF), SALAD, DINOv2 ViT-B/14
#   extra  : CosPlace variants (+torchvision ImageNet backbones), MixVPR (Google Drive),
#            NetVLAD (hloc MATLAB weights), DINOv3 (timm mirrors), VPR-methods-evaluation code
#   anyloc : AnyLoc/DINO hub code, DINOv2 ViT-G/14 weights, global + unstructured vocabularies
# Caches are filled by calling the real loaders, so offline GPU sessions hit exactly the same paths.
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

PROFILE = "__PROFILE__"
W = "/kaggle/working"
TORCH_HOME = f"{W}/torch_home"
HF_HOME = f"{W}/hf_home"
os.environ.update(TORCH_HOME=TORCH_HOME, HF_HOME=HF_HOME, HF_HUB_DISABLE_TELEMETRY="1")
os.makedirs(TORCH_HOME, exist_ok=True)
PINNED = {  # exact upstream commits inspected on 2026-11-06 (HEAD of each default branch)
    "amaralibey/Bag-of-Queries": "1a4965ea7dfd9bd0dd846adf7a0e430f68101d12",
    "gmberton/MegaLoc": "5fe0dd697c4a70ba3e23607f6716ab3c606b16db",
    "facebookresearch/dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "serizba/salad": "6aede13a3f6c25750bf7fde10209c06cb73060bb",
    "AnyLoc/DINO": "eea7d4ca95c132c4ef5e84cd618679d484821945",
    "gmberton/CosPlace": "52b56e95ea62245281281f3bafd7b9390d19a0fd",
    "gmberton/VPR-methods-evaluation": "9c8dfd496b49a49549323b94a6af204cbee91604",
}
# torch.hub cache directory names used by the authors' / upstream loader calls
HUB_DIRS = {"amaralibey/Bag-of-Queries": "amaralibey_bag-of-queries_main", "gmberton/MegaLoc": "gmberton_MegaLoc_main",
            "facebookresearch/dinov2": "facebookresearch_dinov2_main", "serizba/salad": "serizba_salad_main",
            "AnyLoc/DINO": "AnyLoc_DINO_main", "gmberton/CosPlace": "gmberton_cosplace_main"}
REQ = """torch==2.6.0+cu124
torchvision==0.21.0+cu124
triton==3.2.0
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
sympy==1.13.1
mpmath==1.3.0
networkx==3.6.1
Jinja2==3.1.6
MarkupSafe==3.0.3
filelock==3.29.0
fsspec==2026.4.0
typing_extensions==4.15.0
numpy==2.4.4
pillow==12.2.0
huggingface_hub==1.23.0
hf-xet==1.5.1
httpx==0.28.1
httpcore==1.0.9
h11==0.16.0
anyio==4.14.2
certifi==2026.6.17
idna==3.18
click==8.4.2
packaging==26.0
PyYAML==6.0.3
tqdm==4.68.4
safetensors==0.8.0
einops==0.8.2
scipy==1.17.1
opencv-python==5.0.0.93
requests==2.34.2
urllib3==2.7.0
charset-normalizer==3.4.9
setuptools
"""
# not in the authors' freeze but imported by other methods' upstream code (installed with --no-deps)
EXTRA_NODEPS = ["pytorch-lightning==2.6.6", "lightning-utilities", "torchmetrics", "fast-pytorch-kmeans",
                "timm==1.0.30", "psutil", "pytorch-metric-learning==3.0.1"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(cmd, **kw):
    log("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        log(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"command failed: {cmd}")
    return r.stdout


def fetch(url, dest, tries=6):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vprspace-repro/1.0"}), timeout=180) as r, open(dest + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f, 16 << 20)
            os.replace(dest + ".tmp", dest)
            return dest
        except Exception as e:
            log(f"fetch {url} failed ({e!r}), retry {t + 1}")
            time.sleep(2 ** t)
    raise IOError(url)


def pinned_repo(repo, dest_dir):
    """Download the repo at the pinned commit and place it where torch.hub expects its cache."""
    sha = PINNED[repo]
    z = fetch(f"https://github.com/{repo}/archive/{sha}.zip", f"/tmp/{repo.replace('/', '_')}.zip")
    tmp = f"/tmp/unz_{repo.replace('/', '_')}"
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(z).extractall(tmp)
    (inner,) = os.listdir(tmp)
    shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.move(os.path.join(tmp, inner), dest_dir)
    return dest_dir


def hub_dir(repo):
    return os.path.join(TORCH_HOME, "hub", HUB_DIRS[repo])


def wheels():
    wd = f"{W}/wheels"
    os.makedirs(wd, exist_ok=True)
    open(f"{W}/requirements_pinned.txt", "w").write(REQ)
    run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-d", wd, "-r", f"{W}/requirements_pinned.txt",
         "--extra-index-url", "https://download.pytorch.org/whl/cu124"])
    run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "-d", wd] + EXTRA_NODEPS)
    # prove the wheel set is complete: offline install into a clean venv and import
    shutil.rmtree("/tmp/venv_test", ignore_errors=True)
    vpy = "/tmp/venv_test/bin/python"
    try:
        run([sys.executable, "-m", "venv", "/tmp/venv_test"])
        pip = [vpy, "-m", "pip"]
    except RuntimeError:  # no ensurepip in this interpreter: drive the venv with the system pip
        shutil.rmtree("/tmp/venv_test", ignore_errors=True)
        run([sys.executable, "-m", "venv", "--without-pip", "/tmp/venv_test"])
        pip = [sys.executable, "-m", "pip", "--python", vpy]
    run(pip + ["install", "-q", "--no-index", "--find-links", wd, "-r", f"{W}/requirements_pinned.txt"])
    run(pip + ["install", "-q", "--no-index", "--no-deps", "--find-links", wd] + EXTRA_NODEPS)
    out = run([vpy, "-c", "import torch, torchvision, numpy, PIL, scipy, cv2, timm, pytorch_lightning, fast_pytorch_kmeans, pytorch_metric_learning.losses; "
                          "print(torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__, scipy.__version__, cv2.__version__, timm.__version__, pytorch_lightning.__version__)"])
    log("venv test OK:", out.strip())
    return out.strip()


def warm_core():
    import torch
    for repo in ("amaralibey/Bag-of-Queries", "gmberton/MegaLoc", "facebookresearch/dinov2", "serizba/salad"):
        pinned_repo(repo, hub_dir(repo))
    # authors' methods.py calls (source="github" resolves to the cache dirs placed above)
    torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288, trust_repo=True)
    torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)


def warm_extra():
    import torch
    pinned_repo("gmberton/CosPlace", hub_dir("gmberton/CosPlace"))
    for bb, dim in (("ResNet50", 2048), ("ResNet50", 512), ("ResNet101", 2048), ("ResNet152", 2048), ("VGG16", 512)):
        torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=bb, fc_output_dim=dim, trust_repo=True)
    vme = pinned_repo("gmberton/VPR-methods-evaluation", f"{W}/extra/VPR-methods-evaluation")
    for name, url in (("VGG16-NetVLAD-Pitts30K", "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat"),):
        fetch(url, os.path.join(TORCH_HOME, "hub", "netvlad", name + ".mat"))
    import gdown
    md = f"{W}/extra/mixvpr"
    os.makedirs(md, exist_ok=True)
    for fid, fn in (("1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L", "resnet50_MixVPR_4096_channels(1024)_rows(4)"),
                    ("1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj", "resnet50_MixVPR_512_channels(256)_rows(2)"),
                    ("1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz", "resnet50_MixVPR_128_channels(64)_rows(2)")):
        try:
            gdown.download(id=fid, output=os.path.join(md, fn), quiet=True)
        except Exception as e:
            log("mixvpr download failed", fn, repr(e))
    import timm
    for name in ("vit_small_patch16_dinov3.lvd1689m", "vit_base_patch16_dinov3.lvd1689m", "vit_large_patch16_dinov3.lvd1689m"):
        timm.create_model(name, pretrained=True)


def warm_anyloc():
    import torch
    pinned_repo("facebookresearch/dinov2", hub_dir("facebookresearch/dinov2"))
    pinned_repo("AnyLoc/DINO", hub_dir("AnyLoc/DINO"))
    for dom in ("global", "unstructured"):
        torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain=dom, device="cpu", trust_repo=True)


def manifest():
    rows = []
    for base in (TORCH_HOME, HF_HOME, f"{W}/extra", f"{W}/wheels"):
        for dp, _, fs in os.walk(base):
            for fn in fs:
                p = os.path.join(dp, fn)
                if os.path.islink(p):
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for c in iter(lambda: f.read(64 << 20), b""):
                        h.update(c)
                rows.append({"path": os.path.relpath(p, W), "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
    return rows


if __name__ == "__main__":
    t0 = time.time()
    info = {"profile": PROFILE, "python": sys.version, "pinned_commits": PINNED}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fast-pytorch-kmeans", "gdown", "pytorch-metric-learning==3.0.1"], check=False)  # --no-deps: never touch Kaggle's torch
    if "core" in PROFILE:
        info["venv_test"] = wheels()
        warm_core()
    if "extra" in PROFILE:
        warm_extra()
    if "anyloc" in PROFILE:
        warm_anyloc()
    for d in ("/tmp",):
        pass
    info["files"] = manifest()
    info["total_bytes"] = sum(r["bytes"] for r in info["files"])
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(f"{W}/deps_manifest_{PROFILE}.json", "w"), indent=1)
    log(f"done: {len(info['files'])} files, {info['total_bytes'] / 2**30:.2f} GiB, {info['seconds']} s")
