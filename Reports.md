# GPPO Paper-Alignment Report

## A. now matches paper exactly

- Paper-mode topology labels are explicit: `paper_small` and `paper_large`.
- `paper_small` uses `8 RH / 3 ES / 2 RC`.
- `paper_large` uses `64 RH / 4 ES / 2 RC`.
- Paper-mode non-direct links use bandwidth `U(10, 40)` Gbps and latency `U(0, 3.6)` ms.
- Paper-mode direct RH-RC links use per-RH probability `0.10`, bandwidth `160` Gbps, and latency `U(0.1, 0.25)` ms.
- ES and RC capacities are `20` and `100`.
- Request distributions match the paper's eMBB, mMTC, and uRLLC ranges.
- `--paper-mode` uses `288` slots per episode, `600000` timesteps, `32` synchronous environments, and `6` seeds by default.
- GPPO keeps `GINEConv` with two graph convolution layers.
- Paper-mode GNN hidden size is `1024`.
- Policy/value MLP remains two hidden layers of width `256`.
- Xavier uniform initialization is applied to linear layers.

## B. still approximates paper

- Parallel collection is implemented as an in-repo synchronous vectorized rollout path, not Stable-Baselines3 `SubprocVecEnv`.
- Paper-mode reporting aggregates six seeds and per-topology summaries, but exact plotting/table formatting may still differ from the paper.
- Node features include the restored normalized node index in paper mode, alongside the existing semantic node features.

## C. still does not match paper and why

- The resource releasing ratio described in the paper is still not explicitly implemented; requests are still sampled per slot by the current environment.
- The current code still preserves project/debug benchmarks and workflows, so paper-faithful behavior requires using `--paper-mode` or explicit `paper_*` benchmarks.
- Existing old checkpoints are not architecture-compatible with paper-mode GNN input/hidden dimensions; paper-mode writes checkpoint metadata with `checkpoint_family=paper_gppo`.
