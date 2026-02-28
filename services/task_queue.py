"""
task_queue.py — Lightweight background task queue using ThreadPoolExecutor.

No Redis/Celery needed. Uses threads for I/O tasks and multiprocessing
for CPU-heavy work.

Usage:
    from services.task_queue import TaskQueue
    tq = TaskQueue()
    task_id = tq.submit(my_function, arg1, arg2)
    status = tq.get_status(task_id)
"""

import uuid
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock

logger = logging.getLogger("criccric")


class TaskQueue:
    """Simple background task queue backed by ThreadPoolExecutor."""

    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks = {}
        self._lock = Lock()
        logger.info(f"TaskQueue initialised (workers={max_workers})")

    def submit(self, func, *args, **kwargs):
        """
        Submit a function for background execution.

        Returns task_id (str).
        """
        task_id = str(uuid.uuid4())
        future = self.executor.submit(self._run_task, task_id, func, *args, **kwargs)

        with self._lock:
            self._tasks[task_id] = {
                "future": future,
                "submitted_at": time.time(),
                "func_name": func.__name__ if hasattr(func, "__name__") else str(func),
            }

        logger.info(f"Task {task_id} submitted: {self._tasks[task_id]['func_name']}")
        return task_id

    def _run_task(self, task_id, func, *args, **kwargs):
        """Wrapper that logs task start/end."""
        logger.info(f"Task {task_id} started")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"Task {task_id} completed in {elapsed:.1f}s")
            with self._lock:
                self._tasks[task_id]["completed_at"] = time.time()
                self._tasks[task_id]["elapsed"] = elapsed
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Task {task_id} failed after {elapsed:.1f}s: {e}")
            raise

    def get_status(self, task_id):
        """Get current status of a task."""
        with self._lock:
            task = self._tasks.get(task_id)

        if not task:
            return {"status": "not_found", "task_id": task_id}

        future = task["future"]
        info = {
            "task_id": task_id,
            "func": task.get("func_name", ""),
            "submitted_at": task.get("submitted_at", 0),
        }

        if future.running():
            info["status"] = "running"
        elif future.done():
            if future.exception():
                info["status"] = "failed"
                info["error"] = str(future.exception())
            else:
                info["status"] = "completed"
                try:
                    result = future.result(timeout=0)
                    # Only include serialisable results
                    if isinstance(result, (dict, list, str, int, float, bool)):
                        info["result"] = result
                except:
                    pass
            info["elapsed"] = task.get("elapsed", 0)
        else:
            info["status"] = "pending"

        return info

    def list_tasks(self, limit=20):
        """List recent tasks."""
        with self._lock:
            items = sorted(self._tasks.values(),
                          key=lambda x: x.get("submitted_at", 0), reverse=True)

        results = []
        for item in items[:limit]:
            status = self.get_status(
                next(k for k, v in self._tasks.items() if v is item)
            )
            results.append(status)
        return results

    def shutdown(self):
        """Gracefully shut down the executor."""
        self.executor.shutdown(wait=False)
        logger.info("TaskQueue shut down")
