# Probe of a Kaggle L4 (or T4) GPU session: hardware, software and disk, so batch sizes and the
# offline environment can be sized correctly. Internet is OFF for these sessions.
import json, os, platform, shutil, subprocess, sys, time

info = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0))}
try:
    info["mem_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
except Exception:
    pass
for p in ["/kaggle/working", "/kaggle/tmp", "/tmp", "/kaggle/input", "/dev/shm"]:
    try:
        u = shutil.disk_usage(p)
        info[f"disk {p}"] = f"total {u.total / 2**30:.1f} GiB, free {u.free / 2**30:.1f} GiB"
    except Exception as e:
        info[f"disk {p}"] = repr(e)
for cmd in (["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"],
            ["nvidia-smi", "topo", "-m"], ["lscpu"]):
    try:
        info[" ".join(cmd)] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        info[" ".join(cmd)] = repr(e)
for mod in ("torch", "torchvision", "numpy", "PIL", "cv2", "scipy", "sklearn", "timm", "transformers", "xformers",
            "pytorch_lightning", "lightning", "faiss", "huggingface_hub", "safetensors", "einops", "kornia", "gdown"):
    try:
        m = __import__(mod)
        info[mod] = getattr(m, "__version__", "present")
    except Exception:
        info[mod] = None
try:
    import torch
    info["torch_cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["arch_list"] = torch.cuda.get_arch_list()
    info["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    info["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    t = torch.randn(8192, 8192, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        t @ t
    torch.cuda.synchronize()
    info["fp32_matmul_tflops_gpu0"] = round(10 * 2 * 8192**3 / (time.time() - t0) / 1e12, 1)
except Exception as e:
    info["torch_error"] = repr(e)
info["input_dirs"] = sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else None
print(json.dumps(info, indent=1), flush=True)
with open("/kaggle/working/probe.json", "w") as f:
    json.dump(info, f, indent=1)
