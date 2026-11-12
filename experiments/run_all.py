"""Run every experiment (sequentially by default, or in parallel processes with --parallel).
Usage: python experiments/run_all.py [--quick] [--parallel]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp_validation_book.py", "exp_B3_tightness.py", "exp_B2_mu_eff.py", "check_B2_fp_heuristic.py",
           "exp_B1_preconditioning.py", "exp_gamma_delta.py", "exp_A3_lkatyusha_lyapunov.py",
           "exp_A2_naive_acceleration.py", "exp_A4_hard_instance.py", "exp_A4b_instances.py"]
FINAL = ["make_digest.py"]          # run last, after every summary exists


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
    if "--quick" not in sys.argv:
        for s in FINAL:
            subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
    print("all experiments done")


if __name__ == "__main__":
    main()
