# MTP speculative decoding in Rapid-LLM

ECE 201C final project · Jayden Bautista

This project extends UCLA NanoCAD’s Rapid-LLM simulator with a timing and memory model for multi-token prediction (MTP) speculative decoding. It estimates when three draft guesses and one backbone verification pass reduce decode latency, and how much peak GPU memory the extra modules require. The starter is the course’s [Rapid-LLM fork at commit `cd9bd39`](https://github.com/gkarfakis19/Rapid-LLM/tree/cd9bd39dfeafb91cf21e5ee892f1b581d6f9dbdd).

## Repository guide

| Path | What to review |
| --- | --- |
| [Run_Rapid_LLM_MTP_Colab.ipynb](Run_Rapid_LLM_MTP_Colab.ipynb) | Reproduce the main benchmark and zero-guess check in Colab. |
| [source/](source/) | Complete analytical simulator source, with the project changes applied. |
| [presentation/](presentation/) | Slide outline, speaking script, and simulator reports for the baseline and ablations. |

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload [the notebook](Run_Rapid_LLM_MTP_Colab.ipynb).
2. Use a **CPU runtime**. Set `WORKSPACE` in the first code cell to the folder containing the modified `Rapid-LLM/` checkout, then choose **Runtime → Run all**.
3. Read the comparison table in the last cell. The full reports are under `/content/rapid-llm-mtp/results`.

The notebook runs the modified `Rapid-LLM/` checkout already present in the workspace and installs the locked analytical dependencies in its own virtual environment. It runs ordinary decoding, explicit zero guesses, and the three-guess configuration. The A100 is modeled analytically, so the notebook runs without a hardware accelerator or model weights. See the [source README](source/README.md) for local setup.

## What changed

| File | Implementation |
| --- | --- |
| [config.py](source/config.py) | Parses the MTP settings and computes expected emitted tokens from conditional acceptance probabilities. |
| [llm_util.py](source/llm_util.py) | Expands decode matrix shapes from one query to multiple query positions for verification. |
| [inference_timing.py](source/inference_timing.py) | Prices each guess as a projection, transformer block, and shared output-head call. Builds one backbone pass over `D + 1` positions. |
| [simulate_inference_graph.py](source/simulate_inference_graph.py) | Divides draft plus verification time by expected output tokens at each sampled context and integrates the per-token estimates. |
| [memory_estimation.py](source/memory_estimation.py) | Adds the draft projection weights, transformer weights, and KV caches. Reuses the shared weights. |
| [run_perf.py](source/run_perf.py) | Reports the guess count, verification positions, expected outputs, mean decode latency, and peak GPU memory. |
| [MiMo-7B_mtp.yaml](source/configs/model-config/MiMo-7B_mtp.yaml) | Defines the MiMo-7B-shaped workload and the default three guesses with acceptance probabilities `[0.9, 0.8, 0.7]`. |

With three guesses, expected round progress is `1 + 0.9 + 0.9×0.8 + 0.9×0.8×0.7 = 3.124` output tokens. The verifier prices rectangular attention against the longest context in each block. The memory model reserves context-sized draft KV caches and includes in-flight verification positions. The scope is analytical single-GPU timing and peak memory. Draft cache initialization time and temporary draft workspace are outside the modeled costs.

## Results

MiMo-7B-shaped backbone, analytical A100 80 GB, batch size 1, with 1,024 prompt tokens and 1,024 generated tokens.

| Configuration | Decode latency (ms/token) | Peak GPU memory (GiB) |
| --- | ---: | ---: |
| Ordinary decode | 12.33 | 14.63 |
| One guess | 7.02 | 15.02 |
| Two guesses | 5.48 | 15.42 |
| Three guesses, default acceptance | 4.92 | 15.82 |
| Three guesses, low acceptance `[0.1, 0.1, 0.1]` | 13.82 | 15.82 |
| Three guesses, ideal acceptance `[1, 1, 1]` | 3.84 | 15.82 |

The default three-guess case gives **2.51× modeled decode speedup** at a cost of **1.19 GiB** of additional peak memory. The guess-count sweep uses the first `D` entries of the default probability vector. With low acceptance, the round advances just 1.111 tokens in expectation and takes longer per token than ordinary decode.

**Correctness check.** Ordinary decode with the MTP block omitted and explicit `guesses: 0` produced byte-identical timing and memory reports. The [raw reports](presentation/results/) include both runs.

