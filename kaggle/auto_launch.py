"""Launch each account's wave sessions as soon as its vprspace-deps-core is COMPLETE (re-pushes nothing else).
Prints one line per event (for a Monitor) and exits when every session of the plan has been pushed.

python3 kaggle/auto_launch.py kaggle/wave2_plan.json
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from launch import USERNAMES  # noqa: E402
from launch_session import push_session  # noqa: E402

SPLIT = "/tmp/claude-0/-home-user-Mars---VPR-/f4f85d11-307a-560d-8600-f8fcabc2f136/scratchpad/anyloc_demo_split.json"


def status(label, slug):
    r = subprocess.run(["kg", label, "kernels", "status", f"{USERNAMES[label]}/{slug}"], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "?"


def main():
    plan = json.load(open(sys.argv[1]))["sessions"]
    split = json.load(open(SPLIT)) if os.path.exists(SPLIT) else None
    pending = list(range(len(plan)))
    ready = {}
    while pending:
        for k in list(pending):
            s = plan[k]
            lab = s["label"]
            if lab not in ready:
                st = status(lab, "vprspace-deps-core")
                if "COMPLETE" in st:
                    ready[lab] = True
                    print(f"{time.strftime('%H:%M:%S')} deps-core ready for {lab}", flush=True)
                elif "ERROR" in st or "CANCEL" in st:
                    print(f"{time.strftime('%H:%M:%S')} DEPS FAILED for {lab}: {st}", flush=True)
                    ready[lab] = False
                else:
                    continue
            if not ready[lab]:
                pending.remove(k)
                continue
            trajs = split[s["split"]] if ("split" in s and split) else None
            ok = push_session(lab, s["session"], s["gpu"], s["configs"], s["env"], s["deps"], trajs)
            print(f"{time.strftime('%H:%M:%S')} LAUNCHED {lab}/{s['session']} ok={ok}", flush=True)
            pending.remove(k)
        if pending:
            time.sleep(30)
    print("ALL LAUNCHED", flush=True)


if __name__ == "__main__":
    main()
