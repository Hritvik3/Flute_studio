"""
Background job manager.

Runs blocking work (Demucs separation, mastering renders) on worker
threads and exposes progress / logs for polling from the UI.
"""

import threading
import time
import traceback
import uuid
from collections import OrderedDict

from app.config import JOB_LOG_LIMIT


class Job:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.status = "queued"  # queued | running | done | error | cancelled
        self.progress = 0.0
        self.stage = "Waiting..."
        self.log_lines = []
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.finished_at = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # -- worker API ---------------------------------------------------
    def set_progress(self, fraction, stage=None):
        with self._lock:
            self.progress = max(0.0, min(1.0, float(fraction)))
            if stage:
                self.stage = stage

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > JOB_LOG_LIMIT:
                del self.log_lines[: len(self.log_lines) - JOB_LOG_LIMIT]

    def raise_if_cancelled(self):
        if self._cancel.is_set():
            raise JobCancelled()

    def request_cancel(self):
        self._cancel.set()

    @property
    def cancel_requested(self):
        return self._cancel.is_set()

    # -- client API ---------------------------------------------------
    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "progress": round(self.progress, 4),
                "stage": self.stage,
                "log": list(self.log_lines),
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
            }


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, history_limit=100):
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._history_limit = history_limit

    def submit(self, kind, title, target):
        """
        target(job) -> result dict. Runs on a daemon thread.
        """
        job = Job(kind, title)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._history_limit:
                self._jobs.popitem(last=False)

        def runner():
            job.status = "running"
            job.log(f"Started: {title}")
            try:
                job.result = target(job)
                job.status = "done"
                job.progress = 1.0
                job.stage = "Complete"
                job.log("Job finished.")
            except JobCancelled:
                job.status = "cancelled"
                job.stage = "Cancelled"
                job.log("Job cancelled.")
            except Exception as exc:  # noqa: BLE001 - surface any failure to UI
                job.status = "error"
                job.stage = "Failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log("ERROR: " + job.error)
                job.log(traceback.format_exc())
            finally:
                job.finished_at = time.time()

        threading.Thread(target=runner, name=f"job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit=50):
        with self._lock:
            jobs = list(self._jobs.values())[-limit:]
        return [j.to_dict() for j in reversed(jobs)]

    def cancel(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        if job.status in ("queued", "running"):
            job.request_cancel()
            job.log("Cancel requested.")
        return job


JOBS = JobManager()
