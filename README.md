# Post-Transformer Prototype

This repository contains a research prototype for a post-transformer
architecture focused on long-context memory. The central model is a
`CausalPredictiveMemoryMachine` (`CPMM`) that combines persistent slot memory,
energy-based refinement, and a typed graph-memory subsystem for explicit
multi-hop storage and retrieval.

## What is implemented

- A `CPMM` with:
  - persistent latent slots
  - sparse relation mixing between slots
  - chunk-by-chunk event encoding
  - energy-based refinement per chunk
  - typed graph memory for `PASS` / `MAP` / `STEP` events
  - iterative query-time graph walks with learned feedback into the slot state
- A raw-text language milestone with:
  - next-token prediction over synthetic sentence corpora
  - graph-memory supervision on the same sequences
  - answer prediction from memory-conditioned text contexts
- Synthetic long-context benchmarks:
  - passkey retrieval
  - associative recall with alias queries
  - sequential multi-hop tracing
  - harder compositional 3-hop tracing
- Baselines:
  - small decoder-style transformer classifier
  - GRU recurrent classifier
- Ablations:
  - no refinement loop
  - single-slot memory
  - no relation module

## Core update rule

For chunk `c_t`, previous slot state `S_(t-1)`, graph state `G_(t-1)`, and
predicted state `S^pred_t`, the prototype:

1. Encodes local event windows into evidence vectors.
2. Updates a typed graph memory with event-conditioned gains.
3. Feeds graph-query summaries back into the slot writer.
4. Refines the slot state by minimizing:

`E = lambda_obs * E_obs + lambda_dyn * E_dyn + lambda_sparse * E_sparse`

Where:

- `E_obs` measures how well the slots reconstruct the current event evidence.
- `E_dyn` keeps the refined state close to the predicted state.
- `E_sparse` encourages selective slot writing rather than updating everything.

The refinement loop is:

`S^(k+1)_t = S^(k)_t - eta * grad(E(S^(k)_t))`

This keeps the prototype close to the original architecture idea while staying
small enough to train on synthetic tasks.

## Quick start

Create the virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the standard experiment sweep:

```bash
.venv/bin/python run_experiments.py
```

The script writes a JSON summary to `results/`.

Run the harder benchmark sweep:

```bash
.venv/bin/python run_revolutionary_experiments.py
```

This evaluates the same architecture on longer contexts and includes the
compositional multi-hop task.

Run the language-model milestone:

```bash
.venv/bin/python run_language_milestone.py
```

This trains a hybrid graph-language model on raw text-like sequences with
joint next-token, answer, and graph-memory losses.

## Repository layout

- `src/post_transformer_prototype/data.py`: synthetic tasks and vocabulary
- `src/post_transformer_prototype/models.py`: CPMM and baseline models
- `src/post_transformer_prototype/language_train.py`: joint LM + graph-memory milestone training
- `src/post_transformer_prototype/train.py`: training, evaluation, and ablations
- `run_experiments.py`: end-to-end experiment runner
- `run_language_milestone.py`: raw-text language milestone runner
- `run_revolutionary_experiments.py`: harder long-context benchmark runner
