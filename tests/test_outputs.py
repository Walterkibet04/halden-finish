import os
import pwd
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile

import numpy as np
import pytest
from PIL import Image

FINISH = os.environ.get("HALDEN_FINISH", "/app/finish")
HELD = os.environ.get("HALDEN_HELDOUT", "/tests/heldout")
WORKROOT = os.environ.get("HALDEN_WORKROOT", "/var/tmp")
BOUND = 1
RUNNER = "runner"


def _runner():
    """uid/gid of the unprivileged account, or None when not running as root."""
    if os.geteuid() != 0:
        return None
    pw = pwd.getpwnam(RUNNER)
    return pw.pw_uid, pw.pw_gid


def _kill_all_of(uid):
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid == uid:
                os.kill(int(entry), signal.SIGKILL)
        except OSError:
            pass


@pytest.fixture(scope="session", autouse=True)
def _sweep_leftovers():
    yield
    ids = _runner()
    if ids:
        _kill_all_of(ids[0])


def _run_as_runner(src):
    """Copy src to a fresh unguessable path, run the program on it as the
    unprivileged user, and return the finished image bytes' path and workdir."""
    ids = _runner()
    work = tempfile.mkdtemp(prefix="w", dir=WORKROOT)
    os.makedirs(os.path.join(work, "o"))
    inp = os.path.join(work, secrets.token_hex(8) + ".png")
    outp = os.path.join(work, "o", secrets.token_hex(8) + ".png")
    shutil.copy(src, inp)
    kwargs = {}
    if ids:
        for p in (work, os.path.join(work, "o"), inp):
            os.chown(p, ids[0], ids[1])
        os.chmod(work, 0o755)
        kwargs = dict(user=ids[0], group=ids[1], extra_groups=[])
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": work,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    errpath = os.path.join(work, "stderr.txt")
    timed_out = False
    try:
        with open(errpath, "wb") as err:
            try:
                r = subprocess.run([FINISH, inp, outp], stdout=subprocess.DEVNULL, stderr=err,
                                   timeout=180, cwd=work, env=env, start_new_session=True, **kwargs)
            except subprocess.TimeoutExpired:
                timed_out = True
                r = None
    finally:
        if ids:
            _kill_all_of(ids[0])
    return r, timed_out, outp, work


@pytest.mark.parametrize("k", range(10))
def test_heldout_image(k):
    src = os.path.join(HELD, f"{k:02d}_src.png")
    ref = os.path.join(HELD, f"{k:02d}_ref.png")
    assert os.path.exists(FINISH), "/app/finish does not exist"
    r, timed_out, outp, work = _run_as_runner(src)
    try:
        assert not timed_out, "program did not finish within 180 s"
        assert r.returncode == 0, f"program exited {r.returncode}"
        st = os.lstat(outp)
        assert stat.S_ISREG(st.st_mode), "output is not a regular file"
        real_dir = os.path.realpath(os.path.dirname(outp))
        assert real_dir == os.path.realpath(os.path.join(work, "o")), "output directory was redirected"
        im = Image.open(outp)
        want_size = Image.open(src).size
        assert im.mode == "RGB" and im.size == want_size, f"wrong format {im.mode} {im.size}, want {want_size}"
        a = np.asarray(im).astype(np.int16)
        b = np.asarray(Image.open(ref).convert("RGB")).astype(np.int16)
        worst = int(np.abs(a - b).max())
        assert worst <= BOUND, f"max channel difference {worst} exceeds {BOUND}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
