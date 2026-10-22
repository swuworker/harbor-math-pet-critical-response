"""Trusted candidate process boundary. Run only inside a Harbor Docker trial."""
import json
import os
from pathlib import Path
import pwd
import resource
import signal
import subprocess
import tempfile
import time


class CandidateFailure(AssertionError):
    pass


def _drop_privileges():
    account = pwd.getpwnam("candidate")
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def _stop_candidate_processes(uid):
    # Include descendants that deliberately create a new session. This container
    # has exactly one solver UID and verification is sequential, not concurrent.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = next(line for line in (entry / "status").read_text().splitlines()
                        if line.startswith("Uid:"))
            if int(line.split()[1]) == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object field")
        result[key] = value
    return result


def run_payload(payload, budget):
    executable = Path("/opt/venv/bin/python")
    runner = Path("/app/run_case.py")
    if os.geteuid() != 0 or not executable.is_file() or not runner.is_file():
        raise RuntimeError("Trusted verification environment is incomplete")
    account = pwd.getpwnam("candidate")
    if account.pw_uid == 0:
        raise RuntimeError("Candidate account must not be root")
    command = [str(executable), "-I", str(runner)]
    env = {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": account.pw_dir,
           "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    print(json.dumps({"event": "candidate_input", "input": payload}, ensure_ascii=False), flush=True)
    remaining = 10.0 - budget["elapsed"]
    if remaining <= 0:
        raise CandidateFailure("Candidate exceeded the public cumulative 10-second per-item limit")
    # File-size-limited descriptors avoid unbounded communicate() output buffers.
    with tempfile.TemporaryFile(dir="/logs/verifier") as out, tempfile.TemporaryFile(dir="/logs/verifier") as err:
        started = time.monotonic()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                   cwd="/app", env=env, close_fds=True,
                                   start_new_session=True, preexec_fn=_drop_privileges)
        try:
            try:
                process.communicate(json.dumps(payload, allow_nan=False).encode(), timeout=max(0.001, remaining-(time.monotonic()-started)))
            except subprocess.TimeoutExpired as exc:
                raise CandidateFailure("Candidate exceeded the public cumulative 10-second per-item limit") from exc
            elapsed = time.monotonic() - started
            budget["elapsed"] += elapsed
            budget["calls"] += 1
            print(json.dumps({"event": "candidate_budget", "call_seconds": elapsed,
                              "cumulative_seconds": budget["elapsed"], "calls": budget["calls"]}), flush=True)
            if budget["elapsed"] > 10.0:
                raise CandidateFailure("Candidate exceeded the public cumulative 10-second per-item limit")
            out.seek(0)
            stdout = out.read(8 * 1024 * 1024 + 1)
            err.seek(0, os.SEEK_END)
            error_size = err.tell()
            err.seek(max(0, error_size - 3000))
            stderr = err.read().decode("utf-8", errors="replace")
            if process.returncode != 0:
                print(json.dumps({"event": "candidate_process_failure", "returncode": process.returncode,
                                  "stderr_tail": stderr, "stdout_prefix": stdout[:3000].decode("utf-8", errors="replace")}), flush=True)
                raise CandidateFailure(f"Candidate exited {process.returncode}")
            if len(stdout) > 8 * 1024 * 1024:
                raise CandidateFailure("Candidate output exceeds the public 8 MiB limit")
            try:
                result = json.loads(stdout.decode("utf-8"), object_pairs_hook=_unique_object,
                                    parse_int=_json_integer,
                                    parse_float=float,
                                    parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
                print(json.dumps({"event": "candidate_invalid_json", "stdout_prefix": stdout[:3000].decode("utf-8", errors="replace")}), flush=True)
                raise CandidateFailure("Candidate did not return one valid JSON object") from exc
            try:
                print(json.dumps({"event": "candidate_output", "output": result}, ensure_ascii=False,
                                 allow_nan=False), flush=True)
            except (ValueError, TypeError, RecursionError) as exc:
                raise CandidateFailure("Candidate JSON cannot be represented within the public output schema") from exc
            return result
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _stop_candidate_processes(account.pw_uid)
            if process.poll() is None:
                process.wait(timeout=2)


def _json_integer(token):
    if len(token.lstrip("-")) > 4096:
        raise ValueError("Integer token exceeds public 4096-digit limit")
    return int(token)
