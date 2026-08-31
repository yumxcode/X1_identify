import sys, os, subprocess, shutil
print("py", sys.version.split()[0], flush=True)
print("conda:", shutil.which("conda"), flush=True)
if shutil.which("conda"):
    r = subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "pinocchio"],
                       capture_output=True, text=True, timeout=1200)
    print("conda rc:", r.returncode, flush=True)
    print((r.stdout or "")[-200:], (r.stderr or "")[-300:], flush=True)
r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "30", "pin==2.6.9"],
                    capture_output=True, text=True, timeout=900)
print("pip pin==2.6.9 rc:", r2.returncode, flush=True)
if r2.returncode: print((r2.stderr or "")[-200:], flush=True)
try:
    import pinocchio as pin
    print("PINOCCHIO OK", pin.__version__, flush=True)
    m = pin.buildModelFromUrdf("X1_identify/urdf/x1.urdf", pin.JointModelFreeFlyer())
    print("URDF OK nv", m.nv, flush=True)
except Exception as e:
    print("pinocchio FAIL:", e, flush=True)
