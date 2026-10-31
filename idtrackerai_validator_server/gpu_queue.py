"""Queue of GPU-demanding jobs (imgstore-crop-arenas, sleap-train): each waits until the GPU
has room for it, so that jobs do not die of CUDA_ERROR_OUT_OF_MEMORY (what made
imgstore-crop-arenas' NVENC encoders fail and retry their chunk an hour later).

Criterion: GPU memory. The head of the queue starts when

    estimate <= total - used - reserved - MARGIN_MB

`used` is what nvidia-smi reports now (every process, also those outside the queue: nextflow's
SLEAP inference, jobs started by hand). `reserved` is, for every GPU job running (queued ones
and those of a registered kind found running outside the queue), its estimate minus what its
processes use now: a crop's encoders come and go chunk by chunk and a job just started has not
allocated yet, so their room is kept for them. A kind's estimate is its `memory_mb`, or the
peak measured in its last LEARN_RUNS runs (+ 10 %) when higher. A kind may also cap how many of
its jobs run at once (`max_running`). Strict FIFO: when the head does not fit nothing behind it
starts (a large job is not starved by small ones); "Run next" moves an entry to the front. One
job starts per tick (TICK_S), so the next decision sees its memory.

State is one JSON file ($GPU_QUEUE_FILE), read and written under an flock, so every server
instance can enqueue and show it; only the instance holding $GPU_QUEUE_FILE.scheduler (flock
for its lifetime) starts jobs. Jobs run in their own session (they survive server restarts)
under `sh -c` that appends "[exit code N]" to their log, and are found again by pid + start
time. With no server running nothing starts.

A kind is registered with register_kind(name, label, memory_mb, launch, ...):
    launch(entry) -> pid of the session leader (the sh wrapper)
    progress(entry) -> {"done", "total", "unit", ...} or None  (may set entry["info"])
    external() -> [{"pid", "what", "started", "params"}] running outside the queue
    pause(job, on), pause_state(job) -> None (this run cannot be paused) / "running" / "pausing" /
        "paused": optional, for kinds whose runs can stop between steps (imgstore-crop-arenas:
        the chunks being encoded finish, no new one starts). job is a queue entry or an external
        job. A paused job (nothing of it running any more) reserves no GPU memory.
"""
import fcntl
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid

import psutil
from flask import jsonify, request

from idtrackerai_validator_server.resources import read_gpus

logger = logging.getLogger(__name__)

QUEUE_FILE = os.environ.get("GPU_QUEUE_FILE", os.path.expanduser("~/.cache/idtrackerai-validator/gpu_queue.json"))
MARGIN_MB = float(os.environ.get("GPU_QUEUE_MARGIN_MB", 4096))
TICK_S = 10
LEARN_RUNS = 5
KEEP_FINISHED = 30

KINDS = {}
_scheduler_lock = None      # file object holding the scheduler flock
_started = False


def register_kind(name, label, memory_mb, launch, progress=None, external=None, max_running=None,
                  pause=None, pause_state=None):
    KINDS[name] = {"label": label, "memory_mb": memory_mb, "launch": launch, "progress": progress,
                   "external": external, "max_running": max_running, "pause": pause, "pause_state": pause_state}


def _pause_state(kind, job):
    hook = KINDS.get(kind, {}).get("pause_state")
    if hook is None:
        return None
    try:
        return hook(job)
    except Exception as error:
        logger.debug("gpu_queue: pause state of %s: %s", job.get("what"), error)
        return None


# ---- state file ----

class _State:
    """with _State() as state: state["entries"] ... (read under an exclusive flock, written back
    on exit when changed)."""

    def __enter__(self):
        os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
        self._lock = open(QUEUE_FILE + ".lock", "a")
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            with open(QUEUE_FILE) as handle:
                self.state = json.load(handle)
        except (FileNotFoundError, ValueError):
            self.state = {}
        self.state.setdefault("entries", [])
        self._before = json.dumps(self.state, sort_keys=True)
        return self.state

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and json.dumps(self.state, sort_keys=True) != self._before:
                tmp = QUEUE_FILE + ".tmp"
                with open(tmp, "w") as handle:
                    json.dump(self.state, handle, indent=1)
                os.replace(tmp, QUEUE_FILE)
        finally:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()


def read_state():
    with _State() as state:
        return json.loads(json.dumps(state))


# ---- processes ----

def spawn(cmd, cwd, log_path, env=None, header=None):
    """cmd in its own session under sh -c, output to log_path, "[exit code N]" appended."""
    line = " ".join(shlex.quote(part) for part in cmd)
    with open(log_path, "w") as log:
        log.write((header or f"$ cd {shlex.quote(cwd)} && {line}") + "\n")
        log.flush()
        process = subprocess.Popen(["sh", "-c", f'{line}; rc=$?; echo; echo "[exit code $rc]"'], cwd=cwd,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
    return process.pid


def _process(entry):
    """The entry's process if it still runs (same pid and start time)."""
    try:
        process = psutil.Process(entry["pid"])
        if abs(process.create_time() - entry.get("create_time", 0)) < 1 and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except (psutil.Error, KeyError, TypeError):
        pass
    return None


def _tree(pid):
    try:
        process = psutil.Process(pid)
        return {pid} | {child.pid for child in process.children(recursive=True)}
    except psutil.Error:
        return set()


def exit_code(log_path):
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4096))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return None
    lines = [line for line in tail.replace("\r", "\n").split("\n") if line.startswith("[exit code ")]
    if not lines:
        return None
    digits = "".join(c for c in lines[-1] if c.isdigit())
    return int(digits) if digits else None


def stop_pid(pid):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            psutil.Process(pid).terminate()
        except psutil.Error:
            pass


# ---- decisions ----

def estimate_mb(kind, entries, total=None):
    """The kind's memory_mb, or the peak of its last LEARN_RUNS finished runs + 10 % when higher;
    never more than the GPU can give."""
    spec = KINDS.get(kind) or {}
    peaks = [e["peak_mb"] for e in entries if e["kind"] == kind and e["status"] == "done" and e.get("peak_mb")]
    value = max([spec.get("memory_mb") or 0] + [1.1 * p for p in peaks[-LEARN_RUNS:]])
    if total:
        value = min(value, total - MARGIN_MB)
    return value


def _budget(state, gpu, running):
    """running: [(kind, current MiB, pause state)] of every GPU job of a registered kind; a paused
    one (its last steps finished) reserves nothing."""
    entries = state["entries"]
    total, used = gpu["mem_total"], gpu["mem_used"]
    reserved = sum(max(0.0, estimate_mb(kind, entries, total) - (mem or 0))
                   for kind, mem, paused in running if paused != "paused")
    return {"total_mb": total, "used_mb": used, "reserved_mb": reserved, "margin_mb": MARGIN_MB,
            "available_mb": total - used - reserved - MARGIN_MB, "util": gpu.get("util"), "t": time.time()}


def _fmt(mb):
    return f"{mb / 1024:.1f} GB"


def tick(launch=True):
    """Refresh the running entries and their GPU memory; start the head of the queue when it fits."""
    gpus = read_gpus()
    gpu = gpus[0] if gpus else None
    mem_of = {}
    if gpu:
        for p in gpu["processes"]:
            mem_of[p["pid"]] = mem_of.get(p["pid"], 0) + (p["mem"] or 0)
    with _State() as state:
        entries = state["entries"]
        now = time.time()
        queue_pids = set()
        running = []
        for entry in entries:
            if entry["status"] != "running":
                continue
            if _process(entry) is None:
                code = exit_code(entry.get("log"))
                status = "done" if code == 0 else "cancelled" if entry.get("stopped") else "failed"
                entry.update(status=status, ended=now, exit_code=code, mem_mb=None)
                continue
            pids = _tree(entry["pid"])
            queue_pids |= pids
            entry["mem_mb"] = sum(mem_of.get(pid, 0) for pid in pids)
            entry["peak_mb"] = max(entry.get("peak_mb") or 0, entry["mem_mb"])
            progress = KINDS.get(entry["kind"], {}).get("progress")
            if progress:
                try:
                    entry["progress"] = progress(entry)
                except Exception as error:
                    logger.debug("gpu_queue: progress of %s: %s", entry["id"], error)
            entry["pause_state"] = _pause_state(entry["kind"], entry)
            running.append((entry["kind"], entry["mem_mb"], entry["pause_state"]))

        external = []
        for name, spec in KINDS.items():
            if not spec.get("external"):
                continue
            try:
                found = spec["external"]()
            except Exception:
                logger.exception("gpu_queue: external %s", name)
                continue
            for job in found:
                if job["pid"] in queue_pids:
                    continue
                pids = _tree(job["pid"])
                if pids & queue_pids:
                    continue
                mem = sum(mem_of.get(pid, 0) for pid in pids)
                paused = _pause_state(name, job)
                external.append({**job, "kind": name, "mem_mb": mem, "pause_state": paused})
                running.append((name, mem, paused))
        state["external"] = external

        state["gpu"] = _budget(state, gpu, running) if gpu else None
        state["blocked"] = None
        queued = [e for e in entries if e["status"] == "queued"]
        if queued and launch:
            head = queued[0]
            spec = KINDS.get(head["kind"])
            if spec is None:
                state["blocked"] = f"Unknown kind {head['kind']} (not registered in this server)"
            elif gpu is None:
                state["blocked"] = "No GPU visible (nvidia-smi)"
            else:
                need = estimate_mb(head["kind"], entries, gpu["mem_total"])
                n_kind = sum(1 for kind, _, _ in running if kind == head["kind"])
                budget = state["gpu"]
                if spec["max_running"] is not None and n_kind >= spec["max_running"]:
                    state["blocked"] = f"{spec['label']}: {n_kind} running, at most {spec['max_running']} at once"
                elif need > budget["available_mb"]:
                    state["blocked"] = (f"Needs {_fmt(need)}, {_fmt(max(0, budget['available_mb']))} available "
                                        f"({_fmt(budget['total_mb'])} − {_fmt(budget['used_mb'])} used − "
                                        f"{_fmt(budget['reserved_mb'])} reserved for running jobs − {_fmt(MARGIN_MB)} margin)")
                else:
                    try:
                        pid = spec["launch"](head)
                        head.update(status="running", pid=pid, started=now, estimate_mb=need,
                                    create_time=psutil.Process(pid).create_time(), peak_mb=0, mem_mb=0)
                        logger.info("gpu_queue: started %s %s (pid %s)", head["kind"], head["what"], pid)
                    except Exception as error:
                        logger.exception("gpu_queue: launch %s", head["id"])
                        head.update(status="failed", ended=now, error=str(error))
        for entry in queued:
            entry["estimate_mb"] = estimate_mb(entry["kind"], entries, gpu["mem_total"] if gpu else None)

        finished = [e for e in entries if e["status"] in ("done", "failed", "cancelled")]
        drop = {id(e) for e in finished[:-KEEP_FINISHED]}
        state["entries"] = [e for e in entries if id(e) not in drop]
        state["ticked"] = now
        return json.loads(json.dumps(state))


# ---- API for the kinds' modules ----

def enqueue(kind, what, key, params, log):
    """Adds a job (409-style ValueError when one with this kind + key is queued or running)."""
    with _State() as state:
        for entry in state["entries"]:
            if entry["kind"] == kind and entry["key"] == key and entry["status"] in ("queued", "running"):
                raise FileExistsError(f"{KINDS[kind]['label']} is already {entry['status']} for {what}")
        entry = {"id": uuid.uuid4().hex[:8], "kind": kind, "what": what, "key": key, "params": params,
                 "log": log, "status": "queued", "submitted": time.time()}
        state["entries"].append(entry)
    _wake.set()
    return entry


def find(kind, key):
    """The queued or running entry of kind + key, with its position among the queued ones."""
    state = read_state()
    queued = [e for e in state["entries"] if e["status"] == "queued"]
    for entry in state["entries"]:
        if entry["kind"] == kind and entry["key"] == key and entry["status"] in ("queued", "running"):
            if entry["status"] == "queued":
                position = queued.index(entry)
                entry["position"] = position + 1
                entry["reason"] = state.get("blocked") if position == 0 else "Waiting for the jobs ahead"
            return entry
    return None


def cancel(entry_id):
    """Removes a queued entry or stops a running one."""
    with _State() as state:
        for entry in state["entries"]:
            if entry["id"] != entry_id:
                continue
            if entry["status"] == "queued":
                entry.update(status="cancelled", ended=time.time())
                return entry
            if entry["status"] == "running":
                if _process(entry) is not None:
                    stop_pid(entry["pid"])
                entry["stopped"] = True
                return entry
            raise ValueError(f"{entry['what']} is {entry['status']}")
    raise KeyError(entry_id)


def set_paused(job, on):
    """Pauses / resumes a running queue entry or external job (ValueError if it cannot be)."""
    hook = KINDS.get(job["kind"], {}).get("pause")
    if hook is None:
        raise ValueError(f"{KINDS.get(job['kind'], {}).get('label', job['kind'])} jobs cannot be paused")
    if _pause_state(job["kind"], job) is None:
        raise ValueError(f"{job['what']} cannot be paused (started by a version without pause: stop it and resume it)")
    hook(job, on)
    state = _pause_state(job["kind"], job)
    _wake.set()
    return state


# ---- scheduler ----

_wake = threading.Event()


def _scheduler():
    while True:
        try:
            tick()
        except Exception:
            logger.exception("gpu_queue: tick")
        _wake.wait(TICK_S)
        _wake.clear()


def start():
    """Starts the scheduler unless another server instance holds it."""
    global _scheduler_lock, _started
    if _started:
        return True
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    handle = open(QUEUE_FILE + ".scheduler", "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        logger.info("gpu_queue: another server instance schedules the GPU queue")
        return False
    _scheduler_lock, _started = handle, True
    threading.Thread(target=_scheduler, name="gpu-queue", daemon=True).start()
    return True


def register_gpu_queue(app):
    scheduling = start()

    @app.route("/api/gpu_queue", methods=["GET"])
    def gpu_queue_state():
        """entries (queued in order, running, last finished), gpu budget, why the head waits,
        jobs of the kinds running outside the queue, kinds with their estimates."""
        state = read_state()
        total = (state.get("gpu") or {}).get("total_mb")
        kinds = {name: {"label": spec["label"], "estimate_mb": estimate_mb(name, state["entries"], total),
                        "max_running": spec["max_running"]} for name, spec in KINDS.items()}
        return jsonify({**state, "kinds": kinds, "scheduling": scheduling, "tick_s": TICK_S, "now": time.time()})

    @app.route("/api/gpu_queue/<entry_id>/cancel", methods=["POST"])
    def gpu_queue_cancel(entry_id):
        try:
            entry = cancel(entry_id)
        except KeyError:
            return jsonify({"error": f"No queue entry {entry_id}"}), 404
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        _wake.set()
        return jsonify(entry)

    @app.route("/api/gpu_queue/<entry_id>/pause", methods=["POST"])
    def gpu_queue_pause(entry_id):
        """{"paused": bool}: pauses (its current steps finish, no new one starts) or resumes a
        running entry; or, with entry_id external-<pid>, a job running outside the queue."""
        on = bool((request.get_json(silent=True) or {}).get("paused", True))
        state = read_state()
        if entry_id.startswith("external-"):
            pid = int(entry_id.split("-", 1)[1]) if entry_id.split("-", 1)[1].isdigit() else None
            job = next((j for j in state.get("external") or [] if j["pid"] == pid), None)
        else:
            job = next((e for e in state["entries"] if e["id"] == entry_id and e["status"] == "running"), None)
        if job is None:
            return jsonify({"error": f"No running job {entry_id}"}), 404
        try:
            pause_state = set_paused(job, on)
        except ValueError as error:
            return jsonify({"error": str(error)}), 409
        return jsonify({"id": entry_id, "pause_state": pause_state})

    @app.route("/api/gpu_queue/<entry_id>/front", methods=["POST"])
    def gpu_queue_front(entry_id):
        """Moves a queued entry to the head of the queue."""
        with _State() as state:
            entry = next((e for e in state["entries"] if e["id"] == entry_id and e["status"] == "queued"), None)
            if entry is None:
                return jsonify({"error": f"No queued entry {entry_id}"}), 404
            state["entries"].remove(entry)
            first = next((i for i, e in enumerate(state["entries"]) if e["status"] == "queued"), len(state["entries"]))
            state["entries"].insert(first, entry)
        _wake.set()
        return jsonify(entry)
