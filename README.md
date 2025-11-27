# azure-openai-blaster

[![Python versions](https://img.shields.io/pypi/pyversions/azure_openai_blaster.svg)](https://pypi.org/project/azure_openai_blaster/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)

> High-throughput, multi-endpoint scheduler & resilience layer for Azure OpenAI (queueing, rate-limit backoff, endpoint health, weighted routing).

`azure-openai-blaster` lets you fan out chat completion traffic across multiple Azure OpenAI deployments while smoothing rate limits, backing off on transient errors, and auto-disabling unhealthy endpoints — all behind a simple, OpenAI-like API.

---

## ✨ Features

- **Multi-endpoint routing**: Weighted round-robin across any number of deployments.
- **Automatic cooldown & backoff**: Exponential backoff for transient timeouts; header/message–derived cooldown for rate limits.
- **Endpoint health tracking**: Consecutive transient failures trigger auto-disable (with reason preserved).
- **Unified sync / future API**: `chat_completion()` (blocking) or `submit_chat_completion()` (returns `Future[str]`).
- **Streaming support**: Pass `stream=True` to assemble a streamed completion into a final string transparently.
- **Flexible auth**: API key or credential-based (`default`, `az` CLI, or `interactive` browser) selection per deployment.
- **Structured error stats**: Snapshot endpoint state via `AzureEndpointState.report()`.
- **Minimal dependencies**: Only `openai` + `azure-identity`.
- **Config-first**: Simple JSON/YAML→dict config to spin up workers fast.
- **Threaded workers**: Background queue; specify worker count for throughput.

---

## 📦 Installation

```bash
pip install azure_openai_blaster
```

Requires Python ≥ 3.11.

---

## 🚀 Quick Start

```python
from azure_openai_blaster import AzureLLMBlaster
import concurrent.futures

config = {
  "strategy": "weighted",  # currently only 'weighted' implemented
  "deployments": [
    {
      "name": "gpt-4o",
      "endpoint": "https://my-aoai-resource.openai.azure.com/",
      "api_key": "YOUR_KEY",          # or "default" / "az" / "interactive"
      "model": "gpt-4o",
      "weight": 2,
      "temperature": 0.2,
      "max_completion_tokens": 512
    },
    {
      "name": "gpt-4o-backup",
      "endpoint": "https://my-aoai-resource-2.openai.azure.com/",
      "api_key": "default",
      "model": "gpt-4o-mini",
      "weight": 1
    }
  ],
  # Optional runtime overrides:
  # "num_workers": 16,
  # "max_job_retry": 5,
  # "worker_polling_interval": 0.5
}

blaster = AzureLLMBlaster.from_config(config, num_workers=12)

messages = [
  {"role": "system", "content": "You are concise."},
  {"role": "user", "content": "Summarize why backoff matters."},
]

# Blocking call (single request)
text = blaster.chat_completion(messages, temperature=0.1)
print(text)

# Or use the built-in future API (submit + later wait)
future = blaster.submit_chat_completion(messages)
result = future.result(timeout=30)

# Blocking calls executed concurrently via ThreadPoolExecutor
prompts = [
  "Summarize why backoff matters.",
  "Explain weighted round robin scheduling.",
  "List reasons endpoints get temporarily disabled.",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # choose desired concurrency
  futures = [
    executor.submit(
      blaster.chat_completion,
      [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": prompt},
      ],
      temperature=0.1,
    )
    for prompt in prompts
  ]
  for f in concurrent.futures.as_completed(futures):
    print(f.result())


blaster.close()
```

---

## 🧩 Config Schema

Each entry in `deployments` maps to `AzureDeploymentConfig`:

| Field | Required | Description |
| ----- | -------- | ----------- |
| `name` | yes | Identifier for logs/metrics |
| `endpoint` | yes | Base Azure OpenAI endpoint URL |
| `api_key` | yes | Key string or `"default"`, `"az"`, `"interactive"` for credential auth |
| `model` | yes | Deployed model name |
| `api_version` | no (default: `2025-01-01-preview`) | API version |
| `weight` | no (default: 1) | Weighted share in round-robin |
| `temperature` | no (default: 1.0) | Per-endpoint temperature |
| `max_completion_tokens` | no | Cap on generation size |
| `rpm_limit` | future | Reserved; not enforced yet |
| `tpm_limit` | future | Reserved; not enforced yet |

Top-level optional fields (fallback to constructor defaults):
`num_workers`, `max_job_retry`, `worker_polling_interval`.

`strategy` is reserved; currently only weighted round-robin scheduler is active.

---

## 🔄 Scheduling & Resilience

- **Weighted Round Robin**: Endpoint appears in internal ring `weight` times; random initial shuffle.
- **Cooldown Handling**: On `RateLimitError`, parses `Retry-After` header or message (fallback 15s); endpoint excluded until timestamp passes.
- **Transient Failures**: `APITimeoutError` triggers exponential backoff: `base * 2^(failure_streak-1)`.
- **Auto-Disable**: After N consecutive transient failures (`auto_disable_threshold=5`), endpoint disabled with reason.
- **Retry Logic**: Jobs retried up to `max_job_retry` if marked retryable; otherwise exception surfaces via the future/result.

---

## 📡 Streaming

```python
text = blaster.chat_completion(messages, stream=True)
```

Internally collects streamed deltas into a single string. (Incremental callback API not yet implemented.)

---

## 🧪 Advanced Usage

Direct programmatic setup (bypass config dict):

```python
from azure_openai_blaster import (
  AzureLLMBlaster, AzureDeploymentConfig, build_endpoint_states
)

cfgs = [
  AzureDeploymentConfig(
    name="primary",
    endpoint="https://...",
    api_key="default",
    model="gpt-4o",
    weight=3,
    temperature=0.2,
  ),
  AzureDeploymentConfig(
    name="backup",
    endpoint="https://...",
    api_key="YOUR_KEY",
    model="gpt-4o-mini",
    weight=1,
  ),
]

states = build_endpoint_states({"deployments": [c.__dict__ for c in cfgs]})
blaster = AzureLLMBlaster(endpoints=states, num_workers=10)
```

Inspect endpoint health:

```python
for state in states:
  print(state.report())
```

---

## ⚠️ Limitations / Roadmap

- `rpm_limit` / `tpm_limit` not enforced yet.
- Single scheduling strategy.
- No async interface (threaded only).
- No partial-stream callback surface.
- No metrics export integration (you can poll `.report()` manually).

---

## 🧪 Testing / Dev

```bash
git clone https://github.com/jinu-jang/aoai-blaster
cd aoai-blaster
pip install -e ".[dev]"
```

---

## 🤝 Contributing

Pre-alpha; feedback & PRs welcome.

1. Fork & branch
2. Add/adjust tests
3. Maintain formatting (`black`, `isort`)
4. Conventional commits preferred

---

## 📄 License

MIT © 2025 Jinu Jang.
