"""
resources.py  —  how busy this computer is (RAM, CPUs, GPUs), for the Monitor tab's
"Server resources" card.

A daemon thread samples every SAMPLE_S seconds (psutil for CPU and memory, nvidia-smi
for the GPUs) and keeps the last HISTORY_S seconds, so the card shows the recent past
even when it was not open. GET /api/monitor/resources returns the latest sample with
its details (per-core usage, GPU processes) and the busiest processes, plus the history.
"""
import os
import time
import logging
import threading
import subprocess
from collections import deque

import psutil
from flask import jsonify

logger = logging.getLogger(__name__)

SAMPLE_S = 5
HISTORY_S = 3600
# processes returned (the union of the busiest by CPU, by RAM and every GPU process)
TOP_PROCESSES = 12
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")

GPU_FIELDS = ["index", "uuid", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit"]

_history = deque(maxlen=HISTORY_S // SAMPLE_S)
_latest = {}
_lock = threading.Lock()
_started = False


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None    # "[N/A]", "[Not Supported]"


def _nvidia_smi(args):
    out = subprocess.run([NVIDIA_SMI, *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def read_gpus():
    """Per GPU utilization, memory (MiB), temperature, power and its compute processes;
    [] when there is no nvidia-smi or no GPU."""
    try:
        rows = _nvidia_smi([f"--query-gpu={','.join(GPU_FIELDS)}"])
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("resources: nvidia-smi: %s", error)
        return []
    gpus = []
    for row in rows:
        g = dict(zip(GPU_FIELDS, row))
        gpus.append({"index": int(g["index"]), "uuid": g["uuid"], "name": g["name"],
                     "util": _number(g["utilization.gpu"]), "mem_util": _number(g["utilization.memory"]),
                     "mem_used": _number(g["memory.used"]), "mem_total": _number(g["memory.total"]),
                     "temperature": _number(g["temperature.gpu"]),
                     "power": _number(g["power.draw"]), "power_limit": _number(g["power.limit"]),
                     "processes": []})
    try:
        apps = _nvidia_smi(["--query-compute-apps=pid,gpu_uuid,used_memory"])
    except (OSError, subprocess.SubprocessError):
        apps = []
    by_uuid = {g["uuid"]: g for g in gpus}
    for pid, uuid, used in apps:
        if uuid in by_uuid and pid.isdigit():
            by_uuid[uuid]["processes"].append({"pid": int(pid), "mem": _number(used)})
    return gpus


def _sample():
    cores = psutil.cpu_percent(percpu=True)    # since the previous call
    times = psutil.cpu_times_percent()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpus = read_gpus()
    return {
        "t": time.time(),
        "cpu": {"percent": sum(cores) / len(cores) if cores else None, "cores": cores,
                "iowait": getattr(times, "iowait", None),
                "logical": psutil.cpu_count(), "physical": psutil.cpu_count(logical=False),
                "load": os.getloadavg()},
        "memory": {"total": memory.total, "available": memory.available, "used": memory.total - memory.available,
                   "percent": memory.percent, "cache": getattr(memory, "cached", 0) + getattr(memory, "buffers", 0),
                   "swap_total": swap.total, "swap_used": swap.used},
        "gpus": gpus,
    }


def _point(sample):
    """What the history keeps of a sample."""
    return {"t": sample["t"], "cpu": sample["cpu"]["percent"], "mem": sample["memory"]["percent"],
            "gpus": [{"util": g["util"], "mem": 100 * g["mem_used"] / g["mem_total"] if g["mem_total"] else None}
                     for g in sample["gpus"]]}


def _sampler():
    psutil.cpu_percent(percpu=True)
    psutil.cpu_times_percent()
    while True:
        time.sleep(SAMPLE_S)
        try:
            sample = _sample()
        except Exception:
            logger.exception("resources: sample")
            continue
        with _lock:
            _latest.clear()
            _latest.update(sample)
            _history.append(_point(sample))


def start():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_sampler, name="resources-sampler", daemon=True).start()


def _short_command(proc):
    cmdline = proc.info.get("cmdline") or []
    if not cmdline:
        return proc.info.get("name") or ""
    parts = [os.path.basename(cmdline[0])] + cmdline[1:]
    # interpreters: show the script / module rather than the python binary
    if parts[0].startswith("python") and len(parts) > 1:
        rest = parts[1:]
        if rest[0] == "-m" and len(rest) > 1:
            parts = [rest[1]] + rest[2:]
        else:
            parts = [os.path.basename(rest[0])] + rest[1:]
    return " ".join(parts)[:300]


_proc_lock = threading.Lock()


def top_processes(gpu_pids):
    """The busiest processes by CPU (% of one core, since the previous call: psutil keeps
    the Process objects of process_iter between calls) and by RSS, plus every GPU process."""
    rows = []
    with _proc_lock:
        for proc in psutil.process_iter(["pid", "name", "username", "cmdline", "memory_info"]):
            try:
                cpu = proc.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss = proc.info["memory_info"].rss if proc.info.get("memory_info") else 0
            rows.append((proc, cpu, rss))
    keep = {id(r[0]) for r in sorted(rows, key=lambda r: -r[1])[:TOP_PROCESSES]}
    keep |= {id(r[0]) for r in sorted(rows, key=lambda r: -r[2])[:TOP_PROCESSES]}
    keep |= {id(r[0]) for r in rows if r[0].pid in gpu_pids}
    return [{"pid": proc.pid, "user": proc.info.get("username"), "command": _short_command(proc),
             "cpu": cpu, "rss": rss, "gpu_mem": gpu_pids.get(proc.pid)}
            for proc, cpu, rss in rows if id(proc) in keep]


def register_resources(app):
    start()

    @app.route("/api/monitor/resources", methods=["GET"])
    def monitor_resources():
        with _lock:
            latest = dict(_latest)
            history = list(_history)
        if not latest:    # first SAMPLE_S seconds after start
            latest = _sample()
        gpu_pids = {}
        for g in latest["gpus"]:
            for p in g["processes"]:
                gpu_pids[p["pid"]] = gpu_pids.get(p["pid"], 0) + (p["mem"] or 0)
        return jsonify({**latest, "processes": top_processes(gpu_pids), "history": history,
                        "sample_s": SAMPLE_S, "now": time.time()})
