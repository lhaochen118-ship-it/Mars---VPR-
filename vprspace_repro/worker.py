"""Per-GPU descriptor extraction worker. Run with CUDA_VISIBLE_DEVICES set to one GPU.

python worker.py --spec spec.json --worker 0
spec: {"data_root", "desc_dir", "configs": [...], "assign": [[traj, ...] per worker], "torch_home", "vme_dir", "mixvpr_dir"}
"""
import argparse, json, os, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
sys.path.insert(0, str(HERE))
from configs import CONFIGS, outputs_of  # noqa: E402

IMNET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Offline torch.hub: resolve "<owner>/<repo>" to the pre-populated cache directory (pinned commit) instead of probing
# github.com for the default branch. This is exactly torch.hub's own no-internet fallback, made unconditional.
_orig_parse_repo_info = torch.hub._parse_repo_info


def _offline_parse_repo_info(github):
    if ":" not in github:
        owner, name = github.split("/")
        for ref in ("main", "master"):
            if os.path.isdir(os.path.join(torch.hub.get_dir(), f"{owner}_{name}_{ref}")):
                return owner, name, ref
    return _orig_parse_repo_info(github)


torch.hub._parse_repo_info = _offline_parse_repo_info
NUM_WORKERS = int(os.environ.get("VPRS_LOADER_WORKERS", "10"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[w{os.environ.get('VPRS_WORKER', '?')}]", *a, flush=True)


# ---------------------------------------------------------------- preprocessing (CPU, in loader workers)
def dino_v2_resize(x):  # VPR-methods-evaluation ResizingWrapper("dino_v2_resize"): nearest multiple of 14
    h, w = x.shape[-2:]
    return T.functional.resize(x, [round(h / 14) * 14, round(w / 14) * 14], antialias=True)


def anyloc_demo(x, max_img_size=1024):  # AnyLoc demo: cap long side at 1024 (bicubic), center-crop to multiple of 14
    c, h, w = x.shape
    if max(h, w) > max_img_size:
        if h >= w:
            w, h = int(w * max_img_size / h), max_img_size
        else:
            h, w = int(h * max_img_size / w), max_img_size
        x = T.functional.resize(x, (h, w), interpolation=T.InterpolationMode.BICUBIC)
    return T.CenterCrop(((h // 14) * 14, (w // 14) * 14))(x)


def max_side(x, m=640):
    h, w = x.shape[-2:]
    s = m / max(h, w)
    if s >= 1:
        return x
    return T.functional.resize(x, [round(h * s), round(w * s)], interpolation=T.InterpolationMode.BICUBIC, antialias=True)


def make_prep(size):
    norm = T.Normalize(**IMNET)
    if isinstance(size, (tuple, list)):  # identical to the authors' methods.Images transform
        return T.Compose([T.ToTensor(), T.Resize(tuple(size), interpolation=T.InterpolationMode.BICUBIC, antialias=True), norm])
    if size == "native":
        return T.Compose([T.ToTensor(), norm])
    if size == "native14":
        return T.Compose([T.ToTensor(), norm, T.Lambda(dino_v2_resize)])
    if size == "anyloc_demo":
        return T.Compose([T.ToTensor(), norm, T.Lambda(anyloc_demo)])
    if size == "max640":
        return T.Compose([T.ToTensor(), T.Lambda(max_side), norm])
    if size == "gray640":  # DenseVLAD: grayscale, long side 640 (box filter), tensor (1, H, W) in [0, 1]
        def gray(img):
            g = img.convert("L")
            w, h = g.size
            s = 640 / max(w, h)
            if s < 1:
                g = g.resize((round(w * s), round(h * s)), Image.BOX)
            return torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)[None]
        return gray
    if isinstance(size, str) and size.startswith("pilbilinear"):  # e.g. "pilbilinear322" (SALAD eval.py)
        n = int(size[len("pilbilinear"):])
        return T.Compose([T.Resize((n, n), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), norm])
    raise ValueError(size)


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, paths, prep):
        self.paths, self.prep = paths, prep

    def __getitem__(self, i):
        return self.prep(Image.open(self.paths[i]).convert("RGB"))

    def __len__(self):
        return len(self.paths)


# ---------------------------------------------------------------- generic extractor with OOM-adaptive sub-batching
class Extractor:
    def __init__(self, forward, size, loader_bs=64):
        self.forward, self.prep, self.loader_bs, self.sub = forward, make_prep(size), loader_bs, loader_bs

    @torch.no_grad()
    def run_batch(self, x):
        outs, i = [], 0
        while i < x.shape[0]:
            try:
                y = self.forward(x[i:i + self.sub].to(DEV, non_blocking=True))
                outs.append({k: v.float().cpu() for k, v in y.items()} if isinstance(y, dict) else y.float().cpu())
                i += self.sub
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.sub == 1:
                    raise
                self.sub = max(1, self.sub // 2)
                log(f"OOM -> sub-batch {self.sub}")
        if isinstance(outs[0], dict):
            return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
        return torch.cat(outs)

    def __call__(self, paths):
        dl = DataLoader(ImageSet(paths, self.prep), batch_size=self.loader_bs, num_workers=NUM_WORKERS, pin_memory=DEV == "cuda")
        res = [self.run_batch(x) for x in dl]
        if isinstance(res[0], dict):
            return {k: torch.cat([r[k] for r in res]).numpy() for k in res[0]}
        return torch.cat(res).numpy()


# ---------------------------------------------------------------- model builders
def vme_module(name, vme_dir):
    """Import vpr_models.<name> from VPR-methods-evaluation without running its eager package __init__."""
    if "vpr_models" not in sys.modules:
        pkg = types.ModuleType("vpr_models")
        pkg.__path__ = [os.path.join(vme_dir, "vpr_models")]
        sys.modules["vpr_models"] = pkg
    sys.modules.setdefault("gdown", types.ModuleType("gdown"))  # only used for downloading (files are cached)
    import importlib
    return importlib.import_module(f"vpr_models.{name}")


def build(name, spec):
    c = CONFIGS[name]
    k = c["kind"]
    if k == "authors":
        import methods  # the authors' file, unchanged
        m = methods.Method(c["method"])
        if "batch_size" in c:
            m.batch_size = c["batch_size"]
        return lambda paths: m(paths)
    if k == "salad":
        model = torch.hub.load("serizba/salad", "dinov2_salad").eval().to(DEV)
        return Extractor(lambda x: model(x), c["size"], loader_bs=128 if c["size"] != "native14" else 16)
    if k == "anyloc":
        model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain="global", device=DEV)
        cc = torch.load(os.path.join(torch.hub.get_dir(), "checkpoints", "anyloc_files",
                                     "dinov2_vitg14_l31_value_c32_unstructured_c_centers.pt"))
        vlad_u = type(model.vlad)(32)
        vlad_u.c_centers = cc.to(DEV)
        vlad_u.fit(None)
        res = name.split("_", 1)[1]

        def fwd(x):
            feats = model.dino_extractor(x)
            return {f"anyloc_global_{res}": model.vlad.generate_multi(feats),
                    f"anyloc_unstructured_{res}": vlad_u.generate_multi(feats)}
        return Extractor(fwd, c["size"], loader_bs=32 if c["size"] != "anyloc_demo" else 8)
    if k == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=c["backbone"], fc_output_dim=c["dim"]).eval().to(DEV)
        return Extractor(lambda x: model(x), c["size"], loader_bs=64 if c["size"] == "native" else 128)
    if k == "mixvpr":
        mod = vme_module("mixvpr", spec["vme_dir"])
        cwd = os.getcwd()
        os.chdir(spec["mixvpr_root"])  # contains trained_models/mixvpr/<ckpt>
        model = mod.get_mixvpr(descriptors_dimension=c["dim"]).eval().to(DEV)
        os.chdir(cwd)
        return Extractor(lambda x: model(x), c["size"], loader_bs=256)
    if k == "netvlad":
        mod = vme_module("netvlad", spec["vme_dir"])
        model = mod.NetVLAD(descriptors_dimension=c["dim"]).eval().to(DEV)
        return Extractor(lambda x: model(x), c["size"], loader_bs=32)
    if k == "dinov3":
        import timm
        model = timm.create_model(c["arch"], pretrained=True, num_classes=0, dynamic_img_size=True).eval()
        rope = getattr(model, "rope", None)
        if c.get("official_rope", True) and rope is not None and hasattr(rope, "periods"):
            rope.periods = rope.periods.to(torch.bfloat16).to(torch.float32)  # matches official DINOv3 outputs
            log("dinov3: RoPE periods truncated to bfloat16 (official behaviour)")
        model = model.to(DEV)
        return Extractor(lambda x: model.forward_features(x)[:, 0], c["size"], loader_bs=128)
    if k == "densevlad":
        from densevlad_gpu import DenseVLAD
        centers = np.load(os.path.join(spec["desc_dir"], "_densevlad", "vocab_k128.npy"))
        model = DenseVLAD(centers).to(DEV)
        return Extractor(lambda x: model(x), c["size"], loader_bs=8)
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--worker", type=int, required=True)
    a = ap.parse_args()
    os.environ["VPRS_WORKER"] = str(a.worker)
    spec = json.load(open(a.spec))
    trajs = spec["assign"][a.worker]
    root, out = Path(spec["data_root"]), Path(spec["desc_dir"])
    out.mkdir(parents=True, exist_ok=True)
    timing = []
    log(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'} trajectories {len(trajs)}")
    for name in spec["configs"]:
        names = outputs_of(name)
        todo = [t for t in trajs if not all((out / n / f"{t}.npy").exists() for n in names)]
        if not todo:
            continue
        t0 = time.time()
        try:
            ext = build(name, spec)
        except Exception as e:
            log(f"BUILD FAILED {name}: {e!r}")
            timing.append({"config": name, "error": repr(e)})
            continue
        log(f"built {name} in {time.time() - t0:.1f}s")
        for t in todo:
            paths = sorted((root / "images" / t).glob("*.png"))
            t1 = time.time()
            try:
                d = ext(paths)
            except Exception as e:
                log(f"FAILED {name} {t}: {e!r}")
                timing.append({"config": name, "traj": t, "error": repr(e)})
                continue
            d = d if isinstance(d, dict) else {name: d}
            for n, arr in d.items():
                (out / n).mkdir(parents=True, exist_ok=True)
                np.save(out / n / f"{t}.npy", np.ascontiguousarray(arr, dtype=np.float32))
            dt = time.time() - t1
            timing.append({"config": name, "traj": t, "n": len(paths), "sec": round(dt, 2),
                           "dims": {n: list(v.shape) for n, v in d.items()}})
            log(f"{name} {t}: {len(paths)} imgs {dt:.1f}s ({len(paths) / max(dt, 1e-9):.1f} img/s)")
        del ext
        torch.cuda.empty_cache()
    json.dump(timing, open(out / f"_timing_w{a.worker}_{os.environ.get('VPRS_SESSION', '')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
