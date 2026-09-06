import os
import subprocess


def run_job(cmd):
    # eval(cmd) is only a comment
    return subprocess.Popen(cmd, shell=True)


def wipe(path):
    os.system(f"rm -rf {path}")
