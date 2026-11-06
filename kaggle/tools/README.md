# Kaggle helper tools

Tokens are **never** stored in this repository. They live in `/root/.kaggle/accounts.json`
(mode 600) with the shape `{"<label>": {"token": "KGAT_...", "gpu": "L4|T4"}}`.

- `kg <label> <kaggle-cli args...>` — runs the Kaggle CLI (>= 2.2) as the given account
  (exports `KAGGLE_API_TOKEN` for that single process only; nothing is printed).
- `kwatch <label> <owner/slug> [queue_stuck_min=8] [max_runtime_min=180]` — polls a kernel until it
  finishes. Exit codes: 0 complete, 1 error/cancelled, 3 still queued after `queue_stuck_min`
  (procedure: cancel and retry up to 4 times), 4 running longer than `max_runtime_min` (possible hang).
  Kaggle does not expose live logs of a running kernel through the API, so "no output" is judged by
  the queue state.

Install: `pip install kaggle && ln -sf $PWD/kaggle/tools/kg /usr/local/bin/kg && ln -sf $PWD/kaggle/tools/kwatch /usr/local/bin/kwatch`
