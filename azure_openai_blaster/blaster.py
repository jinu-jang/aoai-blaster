import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from azure_openai_blaster.azure_endpoint_state import AzureEndpointState
from azure_openai_blaster.initialization import build_endpoint_states
from azure_openai_blaster.requesting import invoke_endpoint
from azure_openai_blaster.scheduler import WeightedRRScheduler


@dataclass
class _Job:
    """Internal representation of a single chat request."""

    messages: List[ChatCompletionMessageParam]
    """List of chat messages for the request."""
    kwargs: Dict[str, Any]
    """Additional keyword arguments for invoke_endpoint."""
    future: Future[str]
    """Attached future to be resolved with the response content."""

    retry_count: int = 0
    """Count of how many times this job has been retried.
    Used for filtering problematic jobs.
    """


class AzureLLMBlaster:
    """
    Multi-endpoint, multi-worker Azure OpenAI "blaster".

    Public API is intentionally similar to the OpenAI client:

        blaster = AzureLLMBlaster.from_config(config_dict, num_workers=24)
        text = blaster.chat_completion(messages=[...], temperature=0.2)

    The caller does not choose which endpoint or which worker handles
    a request; scheduling and retry/cooldown behavior are handled internally.
    """

    def __init__(
        self,
        endpoints: List[AzureEndpointState],
        num_workers: int = 8,
        max_job_retry: int = 5,
        worker_polling_interval: float = 0.5,
    ):
        """
        Initialize a new AzureLLMBlaster.

        Parameters:
            endpoints: List of AzureEndpointState objects that the blaster can send traffic to. \
                At least one endpoint is required.
            num_workers: Number of background worker threads that will pull jobs from the \
                internal queue and execute them. Defaults to 8.
            max_job_retry: Maximum number of attempts for a single job before \
                it is considered problematic and dropped. Defaults to 5.
            worker_polling_interval: Interval, in seconds, for workers to poll the job queue \
                when idle. Defaults to 0.5.
        """
        if not endpoints:
            raise ValueError("AzureLLMBlaster requires at least one endpoint.")

        self._endpoints = endpoints
        self._scheduler = WeightedRRScheduler(endpoints)
        self._queue: Queue[_Job] = Queue()
        self._num_workers = num_workers
        self._max_job_retry = max_job_retry
        """Mark job as problematic after this many retries."""

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._closed = False
        self._worker_polling_interval = worker_polling_interval
        """Interval (seconds) for workers to poll the job queue."""

        for i in range(num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"azure-llm-blaster-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: dict, num_workers: Optional[int] = None
    ) -> "AzureLLMBlaster":
        """
        Construct a blaster from a JSON-style config dict:

            {
              "strategy": "weighted",
              "deployments": [
                { "name": "...", "endpoint": "...", "api_key": "...", ... },
                ...
              ],

              // Optional overrides for __init__ defaults:
              // "num_workers": 24,
              // "max_job_retry": 5,
              // "worker_polling_interval": 0.5
            }

        Values omitted from the config fall back to the defaults.
        """
        endpoints = build_endpoint_states(config)

        init_kwargs: dict[str, Any] = {}
        if num_workers is not None:
            init_kwargs["num_workers"] = num_workers

            if "num_workers" in config:
                logging.info(
                    "Overriding AzureLLMBlaster num_workers with explicit argument. "
                    f"{config['num_workers']} -> {num_workers}",
                )
        elif "num_workers" in config:
            init_kwargs["num_workers"] = config["num_workers"]

        if "max_job_retry" in config:
            init_kwargs["max_job_retry"] = config["max_job_retry"]
        if "worker_polling_interval" in config:
            init_kwargs["worker_polling_interval"] = config["worker_polling_interval"]

        return cls(endpoints=endpoints, **init_kwargs)

    @classmethod
    def from_config_file(
        cls, json_path: str, num_workers: Optional[int] = None
    ) -> "AzureLLMBlaster":
        """
        Construct a blaster from a JSON config file on disk.
        """
        from json import load

        with open(json_path, "r", encoding="utf-8") as f:
            config = load(f)
        return cls.from_config(config, num_workers=num_workers)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def submit_chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> Future[str]:
        """
        Submit a chat completion request and return a Future.

        Parameters
        ----------
        messages : iterable of ChatCompletionMessageParam
            Standard OpenAI-style chat messages.
        **kwargs :
            Additional keyword arguments forwarded to `invoke_endpoint`,
            e.g. `stream=True` if desired.
        Returns
        -------
        Future[str]
            A Future that will resolve to the response content from the model.

        Raises
        ------
        RuntimeError
            If the blaster has been closed and cannot be used.
        """
        if self._closed:
            raise RuntimeError("AzureLLMBlaster has been closed and cannot be used.")

        future: Future[str] = Future()
        job = _Job(messages=list(messages), kwargs=kwargs, future=future)
        self._queue.put(job)

        # Block until a worker sets result or exception
        return future

    def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> str:
        """
        Submit a chat completion request and block until the result is ready.

        Parameters
        ----------
        messages : iterable of ChatCompletionMessageParam
            Standard OpenAI-style chat messages.
        **kwargs :
            Additional keyword arguments forwarded to `invoke_endpoint`,
            e.g. `stream=True` if desired.

        Returns
        -------
        str
            The response content from the model.

        Raises
        ------
        BaseException
            Propagates non-retryable errors from the underlying endpoint
            (e.g., BadRequestError, AuthenticationError, etc.).
        """
        # Block until a worker sets result or exception
        return self.submit_chat_completion(messages, **kwargs).result()

    def close(self, wait: bool = True) -> None:
        """
        Signal all workers to stop and optionally wait for them.

        This does NOT drain the queue; any pending jobs will never be processed
        once the workers exit. Call this when you are done using the blaster.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()

        if wait:
            for t in self._threads:
                t.join()

    # ---------------------------------------------------------------------
    # Internal worker logic
    # ---------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Worker thread main loop: pull jobs, route to endpoint, handle retry."""
        name = threading.current_thread().name

        while not self._stop.is_set():
            try:
                job: _Job = self._queue.get(timeout=self._worker_polling_interval)
            except Empty:
                logging.debug(f"{name}: no job; checking for stop signal.")
                continue

            try:
                logging.debug(f"{name}: processing job.")
                self._handle_job(job)
            finally:
                # We don't rely on join() semantics here, but keeping this
                # correct is cheap if you ever want to use queue.join().
                logging.debug(f"{name}: job done.")
                self._queue.task_done()

    def _handle_job(self, job: _Job) -> None:
        """Process a single job using the scheduler + invoke_endpoint."""
        # If the future is already resolved (e.g., caller cancelled), skip work.
        if job.future.done():
            return

        # Block until some endpoint is available
        ep = self._scheduler.next()

        result = invoke_endpoint(ep, job.messages, **job.kwargs)

        if result.ok:
            if not job.future.done():
                # `response` is a string (see RequestResult in requesting.py)
                job.future.set_result(result.response or "")
            return

        job.retry_count += 1
        if job.retry_count >= self._max_job_retry:
            # Too many retries: surface to caller via the future.
            if not job.future.done():
                if result.error is not None:
                    job.future.set_exception(result.error)
                else:
                    job.future.set_exception(
                        TimeoutError(
                            f"Job deemed problematic after {job.retry_count} tries."
                        )
                    )
            return

        if result.retryable:
            # Transient error: requeue the same job to be tried again later.
            # Note: we do NOT touch the future here; caller still waits.
            self._queue.put(job)
        else:
            # Non-retryable error: surface to caller via the future.
            if not job.future.done():
                if result.error is not None:
                    job.future.set_exception(result.error)
                else:
                    job.future.set_exception(
                        RuntimeError("LLM request failed without an explicit error.")
                    )
