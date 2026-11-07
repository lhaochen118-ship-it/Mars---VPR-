# Build VLFeat 0.9.21 (official binary release, sha256-pinned) + the phow shim, self-test it, and publish the
# libraries as this kernel's output so internet-off GPU sessions can attach it (kernel_sources). CPU, Internet ON.
import base64, hashlib, json, math, os, shutil, subprocess, sys, tarfile, time, urllib.request

OUT = "/kaggle/working/vlfeat"
URL = "https://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz"
SHA256 = "91a9dd856210473b1eef7098fa84f3a1e1ac020f2a6e97d1df9727354d47a211"
SHIM_C = base64.b64decode("__SHIM_C_B64__").decode()
PHOW_PY = base64.b64decode("__PHOW_PY_B64__").decode()
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
tgz = "/tmp/vlfeat.tgz"
urllib.request.urlretrieve(URL, tgz)
h = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
print("sha256", h, "ok" if h == SHA256 else "MISMATCH", flush=True)
assert h == SHA256
with tarfile.open(tgz) as tf:
    tf.extractall("/tmp/vlsrc", members=[m for m in tf.getmembers()
                                        if m.name.startswith(("vlfeat-0.9.21/vl/", "vlfeat-0.9.21/bin/glnxa64/libvl.so",
                                                              "vlfeat-0.9.21/COPYING", "vlfeat-0.9.21/toolbox/sift/vl_phow.m",
                                                              "vlfeat-0.9.21/toolbox/sift/vl_dsift.c",
                                                              "vlfeat-0.9.21/toolbox/imop/vl_imsmooth.c"))])
V = "/tmp/vlsrc/vlfeat-0.9.21"
shutil.copy(f"{V}/bin/glnxa64/libvl.so", OUT)
for f in ("COPYING", "toolbox/sift/vl_phow.m", "toolbox/sift/vl_dsift.c", "toolbox/imop/vl_imsmooth.c"):
    shutil.copy(f"{V}/{f}", OUT)
open(f"{OUT}/phow_shim.c", "w").write(SHIM_C)
open(f"{OUT}/phow.py", "w").write(PHOW_PY)
subprocess.run(["gcc", "--version"], check=True)
subprocess.run(["gcc", "-O2", "-Wall", "-shared", "-fPIC", f"-I{V}", f"{OUT}/phow_shim.c", f"-L{OUT}", "-lvl",
                "-Wl,-rpath,$ORIGIN", "-o", f"{OUT}/libphow.so"], check=True)
print(subprocess.run(["ldd", f"{OUT}/libphow.so"], capture_output=True, text=True).stdout, flush=True)

# ---------------- self-test (MATLAB conventions) ----------------
os.environ["VPRS_VLFEAT_DIR"] = OUT
sys.path.insert(0, OUT)
import numpy as np
from phow import phow, dsift, imsmooth
rep = {}
H, W = 480, 640
f, d = phow(np.zeros((H, W), np.float32))
rep["flat_all_zero"] = bool((d == 0).all())
# expected frame count per scale: centres from off+1.5*size .. on a step-2 grid inside the image (vl_dsift bounds)
rep["frames_per_scale"] = {int(s): int((f[:, 3] == s).sum()) for s in (4, 6, 8, 10)}
rep["first_centres"] = {int(s): f[f[:, 3] == s][0, :2].tolist() for s in (4, 6, 8, 10)}  # all scales share centres
x = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))      # intensity grows with column (MATLAB x)
y = np.ascontiguousarray(x[:, :H].T[:H, :H])                       # grows with row (MATLAB y), square image
fx, dx = phow(x)
fy, dy = phow(y)
ox = dx.reshape(-1, 16, 8).sum((0, 1)); oy = dy.reshape(-1, 16, 8).sum((0, 1))
rep["ramp_x_orientation_hist"] = ox.tolist()
rep["ramp_y_orientation_hist"] = oy.tolist()
rep["ramp_x_argmax_bin"], rep["ramp_y_argmax_bin"] = int(ox.argmax()), int(oy.argmax())
rng = np.random.default_rng(0)
for name, shape in (("640x480", (480, 640)), ("2064x1544", (1544, 2064))):
    im = rng.random(shape, dtype=np.float32)
    t = time.time(); fr, de = phow(im); rep[f"time_{name}"] = round(time.time() - t, 3)
    rep[f"n_{name}"] = int(len(fr))
    rep[f"contrast_pct_{name}"] = np.percentile(fr[:, 2], [1, 50, 99]).round(4).tolist()
# smoothing must preserve a constant image and the image mean of a random one (normalised kernel, continuity pad)
c = imsmooth(np.full((50, 60), 0.3, np.float32), 1.5)
rep["smooth_const_err"] = float(abs(c - 0.3).max())
print(json.dumps(rep, indent=1), flush=True)
json.dump(rep, open(f"{OUT}/selftest.json", "w"), indent=1)
assert rep["flat_all_zero"] and rep["smooth_const_err"] < 1e-6
print("BUILD OK", round(time.time() - t0, 1), "s")
