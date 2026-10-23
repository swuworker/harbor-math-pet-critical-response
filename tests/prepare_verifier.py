"""Prepare the trusted verifier boundary, invoked by Harbor as root."""
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import subprocess
import sys

if os.geteuid() != 0:
    raise RuntimeError("Verifier must run as root")
account = pwd.getpwnam("candidate")
if account.pw_uid == 0:
    raise RuntimeError("Candidate is privileged")

# Stop processes left by the solver before testing its files in fresh subprocesses.
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        uid_line = next(line for line in (entry / "status").read_text().splitlines() if line.startswith("Uid:"))
        if int(uid_line.split()[1]) == account.pw_uid:
            os.kill(int(entry.name), signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
        pass

for protected in [Path("/tests"), Path("/solution")]:
    if protected.exists():
        if protected.is_symlink():
            raise RuntimeError("Protected verifier path is a symlink")
        os.chown(protected, 0, 0)
        protected.chmod(0o700)
        for path in protected.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("Protected test/solution content contains a symlink")
            os.chown(path, 0, 0)
            path.chmod(0o700 if path.is_dir() else 0o600)
Path("/tests/test.sh").chmod(0o755)

output = Path("/logs/verifier")
if output.is_symlink():
    raise RuntimeError("Verifier output path is a symlink")
output.mkdir(exist_ok=True, parents=True)
os.chown(output, 0, 0)
output.chmod(0o700)
owned = {'reward.txt', 'reward.json', 'case-results.json', 'verification-security.json',
         'verifier-failure.json'}
for path in output.iterdir():
    if not (path.name in owned or path.name.startswith(('case-', '.quarantined-', '.zero-'))):
        continue
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()

def demote():
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)

probe = subprocess.run([sys.executable, "-I", "-c",
    "import json,os;print(json.dumps({'uid':os.getuid(),'tests_readable':os.access('/tests',os.R_OK),"
    "'solution_readable':os.access('/solution',os.R_OK),'reward_writable':os.access('/logs/verifier',os.W_OK)}))"],
    text=True, capture_output=True, check=True, timeout=10, preexec_fn=demote, cwd="/", env={"PATH":"/usr/bin:/bin"})
observed = json.loads(probe.stdout)
if observed["uid"] == 0 or any(observed[key] for key in ("tests_readable", "solution_readable", "reward_writable")):
    raise RuntimeError("Candidate can access protected verification files")
record = {"phase": "verification", "probe": observed,
          "test_sh_mode": oct(stat.S_IMODE(Path('/tests/test.sh').stat().st_mode)),
          "test_sh_uid": Path('/tests/test.sh').stat().st_uid}
(output / "verification-security.json").write_text(json.dumps(record, indent=2) + "\n")
