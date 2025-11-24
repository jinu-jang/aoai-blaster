import random
import threading
import time

from azure_openai_blaster.azure_endpoint_state import AzureEndpointState


class WeightedRRScheduler:
    def __init__(self, endpoints: list[AzureEndpointState]):
        self.endpoints = endpoints
        """List of all endpoints being scheduled."""

        ring = []
        for ep in endpoints:
            ring.extend([ep] * max(1, ep.cfg.weight))
        random.shuffle(ring)

        self._ring: list[AzureEndpointState] = ring
        """Weighted round-robin ring of endpoints."""
        self._idx: int = 0
        """Current index in the round-robin ring."""
        self._lock = threading.Lock()

    def next(self) -> AzureEndpointState:
        """Pick the next available endpoint; if all cooling down, wait minimally."""
        while True:
            with self._lock:
                n = len(self._ring)
                now = time.monotonic()

                available_found = False
                soonest = None
                for _ in range(n):
                    ep = self._ring[self._idx]
                    self._idx = (self._idx + 1) % n
                    if ep.disabled:
                        continue
                    if ep.available(now=now):
                        return ep

                    available_found = True
                    if soonest is None or ep.cooldown_until < soonest:
                        soonest = ep.cooldown_until

                if not available_found or soonest is None:
                    raise RuntimeError(
                        "WeightedRRScheduler failed to schedule next() "
                        "as there were no available endpoints. "
                        "Were they all disabled due to consecutive failures?"
                    )
                delay = max(0.0, soonest - now)

            if delay > 0:
                time.sleep(delay)
