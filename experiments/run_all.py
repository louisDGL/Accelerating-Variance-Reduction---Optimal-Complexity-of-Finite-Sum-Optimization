"""Run every experiment (sequentially by default, or in parallel processes with --parallel).
Usage: python experiments/run_all.py [--quick] [--parallel]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp_B3_tightness.py", "exp_B2_mu_eff.py", "exp_B1_preconditioning.py", "exp_gamma_delta.py",
           "exp_A2_naive_acceleration.py", "exp_A4_hard_instance.py"]


def main():
    args = [a for a in sys.argv[1:] if a != "--parallel"]
    parallel = "--parallel" in sys.argv
    procs = []
    for s in SCRIPTS:
        cmd = [sys.executable, os.path.join(HERE, s)] + args
        print("running", " ".join(cmd), flush=True)
        if parallel:
            procs.append(subprocess.Popen(cmd))
        else:
            subprocess.run(cmd, check=True)
    for p in procs:
        p.wait()
    print("all experiments done")


if __name__ == "__main__":
    main()
