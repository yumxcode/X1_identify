import sys, os, subprocess
print("HELLO-2 py", sys.version, flush=True)
try:
    import numpy as np
    print("numpy", np.__version__, flush=True)
except Exception as e:
    print("numpy FAIL:", e, flush=True)
try:
    import pinocchio as pin
    print("pinocchio", pin.__version__, flush=True)
except Exception as e:
    print("pinocchio FAIL:", e, flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "20", "pin"],
                   capture_output=True, text=True, timeout=300)
print("pip rc:", r.returncode, flush=True)
print("pip tail:", (r.stderr or r.stdout)[-300:], flush=True)
print("CWD:", os.getcwd(), "| ls:", os.listdir("."), flush=True)
