"""
jobs.py  —  long-running background jobs with progress, polled by the frontend console.

A job is a list of items (e.g. one per video being made), each with a progress in
[0, 1] and a status. The work runs in a daemon thread; the frontend polls
GET /api/jobs/<id> and, when the job produced a file, fetches it once from
GET /api/jobs/<id>/download.
"""
import os
import shutil
import logging
import itertools
import threading
from collections import OrderedDict

from flask import jsonify, send_file

logger = logging.getLogger(__name__)

# Finished jobs beyond this many are forgotten, and their files deleted.
MAX_JOBS = 50

_jobs = OrderedDict()
_jobs_lock = threading.Lock()
_ids = itertools.count(1)


class Job:

    def __init__(self, title, items):
        self.id = str(next(_ids))
        self.title = title
        self.status = "running"          # running | done | failed
        self.error = None
        self.items = [{"name": name, "progress": 0.0, "status": "pending", "error": None} for name in items]
        self.directory = None            # where results were saved on the server
        self.result_path = None          # file to hand to the client, if any
        self.download_name = None
        self.workdir = None              # deleted when the job is forgotten / downloaded
        self._lock = threading.Lock()

    def update(self, index, **fields):
        with self._lock:
            self.items[index].update(fields)

    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "title": self.title,
                "status": self.status,
                "error": self.error,
                "items": [dict(item) for item in self.items],
                "directory": self.directory,
                "download": self.result_path is not None,
            }

    def cleanup(self):
        if self.workdir and os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)


def start_job(title, items, work):
    """Run `work(job)` in a background thread. `items` are the names of the units of
    work (one progress bar each). Returns the Job."""
    job = Job(title, items)
    with _jobs_lock:
        _jobs[job.id] = job
        finished = [j for j in _jobs.values() if j.status != "running"]
        for old in finished[:max(0, len(_jobs) - MAX_JOBS)]:
            old.cleanup()
            del _jobs[old.id]

    def run():
        try:
            work(job)
            job.status = "done"
        except Exception as error:
            logger.exception("Job %s (%s) failed", job.id, title)
            job.error = str(error)
            job.status = "failed"

    threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
    return job


def register_jobs(app):

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def get_job(job_id):
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": f"No job {job_id}"}), 404
        return jsonify(job.to_dict())

    @app.route("/api/jobs/<job_id>/download", methods=["GET"])
    def download_job(job_id):
        """The file produced by the job. Available once: it is deleted after this."""
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": f"No job {job_id}"}), 404
        if job.status != "done" or job.result_path is None or not os.path.exists(job.result_path):
            return jsonify({"error": f"Job {job_id} has nothing to download"}), 404
        handle = open(job.result_path, "rb")   # the open handle outlives the deleted file
        job.cleanup()
        job.result_path = None
        return send_file(handle, as_attachment=True, download_name=job.download_name)
