import time
from collections import deque

import numpy as np


class ActionChunkScheduler:
    """Synchronous action chunk queue for rollout command timing."""

    def __init__(
        self,
        action_dim=None,
        refill_threshold=0,
        blend_steps=0,
        max_queue_size=0,
        underrun_policy="hold",
    ):
        self.action_dim = None if action_dim is None else int(action_dim)
        self.refill_threshold = max(int(refill_threshold), 0)
        self.blend_steps = max(int(blend_steps), 0)
        self.max_queue_size = max(int(max_queue_size), 0)
        self.underrun_policy = str(underrun_policy)
        if self.underrun_policy not in ("hold", "error"):
            raise ValueError(
                "ActionChunkScheduler underrun_policy must be 'hold' or 'error'."
            )
        self.reset()

    def reset(self):
        self._queue = deque()
        self._last_action = None
        self.push_count = 0
        self.pop_count = 0
        self.underrun_count = 0
        self.boundary_jump_list = []
        self.queue_depth_list = []
        self.chunk_size_list = []
        self.inference_latency_list = []
        self.obs_to_push_latency_list = []

    @property
    def queue_depth(self):
        return len(self._queue)

    @property
    def last_action(self):
        return None if self._last_action is None else self._last_action.copy()

    def needs_refill(self):
        return len(self._queue) <= self.refill_threshold

    def push(
        self,
        action_chunk,
        obs_timestamp=None,
        inference_start_time=None,
        inference_end_time=None,
    ):
        chunk = self._as_action_chunk(action_chunk)
        if len(chunk) == 0:
            return

        now = time.perf_counter()
        if inference_start_time is not None:
            end_time = now if inference_end_time is None else inference_end_time
            self.inference_latency_list.append(float(end_time - inference_start_time))
        if obs_timestamp is not None:
            self.obs_to_push_latency_list.append(float(now - obs_timestamp))

        queued_chunk = self._make_boundary_blended_chunk(chunk)
        self._queue.extend(action.copy() for action in queued_chunk)
        if self.max_queue_size > 0:
            while len(self._queue) > self.max_queue_size:
                self._queue.pop()

        self.push_count += 1
        self.chunk_size_list.append(int(len(chunk)))
        self.queue_depth_list.append(len(self._queue))

    def next_action(self):
        if self._queue:
            action = self._queue.popleft()
            self._last_action = action.copy()
            self.pop_count += 1
            self.queue_depth_list.append(len(self._queue))
            return action.copy()

        self.underrun_count += 1
        self.queue_depth_list.append(0)
        if self.underrun_policy == "error" or self._last_action is None:
            raise RuntimeError("ActionChunkScheduler queue underrun.")
        return self._last_action.copy()

    def get_statistics(self):
        stats = {
            "queue_depth": self.queue_depth,
            "push_count": self.push_count,
            "pop_count": self.pop_count,
            "underrun_count": self.underrun_count,
        }
        if self.queue_depth_list:
            stats["queue_depth_mean"] = float(np.mean(self.queue_depth_list))
            stats["queue_depth_min"] = int(np.min(self.queue_depth_list))
            stats["queue_depth_max"] = int(np.max(self.queue_depth_list))
        if self.chunk_size_list:
            stats["chunk_size_mean"] = float(np.mean(self.chunk_size_list))
        if self.boundary_jump_list:
            stats["boundary_jump_mean"] = float(np.mean(self.boundary_jump_list))
            stats["boundary_jump_max"] = float(np.max(self.boundary_jump_list))
        if self.inference_latency_list:
            stats["inference_latency_mean"] = float(
                np.mean(self.inference_latency_list)
            )
            stats["inference_latency_max"] = float(np.max(self.inference_latency_list))
        if self.obs_to_push_latency_list:
            stats["obs_to_push_latency_mean"] = float(
                np.mean(self.obs_to_push_latency_list)
            )
            stats["obs_to_push_latency_max"] = float(
                np.max(self.obs_to_push_latency_list)
            )
        return stats

    def _as_action_chunk(self, action_chunk):
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis]
        if chunk.ndim != 2:
            raise ValueError(
                "Action chunk must have shape (action_dim,) or "
                f"(chunk_size, action_dim), got {chunk.shape}."
            )
        if self.action_dim is None:
            self.action_dim = int(chunk.shape[1])
        elif chunk.shape[1] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: expected {self.action_dim}, got {chunk.shape[1]}."
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Action chunk contains NaN or Inf.")
        return chunk.copy()

    def _make_boundary_blended_chunk(self, chunk):
        start_action = self._queue[-1] if self._queue else self._last_action
        if start_action is None:
            return chunk

        boundary_jump = float(np.linalg.norm(chunk[0] - start_action))
        self.boundary_jump_list.append(boundary_jump)
        if self.blend_steps <= 0:
            return chunk

        blend_count = min(self.blend_steps, len(chunk))
        alphas = np.linspace(
            1.0 / float(blend_count),
            1.0,
            blend_count,
            dtype=np.float64,
        )
        blended = chunk.copy()
        blended[:blend_count] = (
            (1.0 - alphas[:, None]) * start_action
            + alphas[:, None] * chunk[:blend_count]
        )
        return blended
