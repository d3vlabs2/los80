# LOS80 Performance and Production Guide

## Optimization strategy

LOS80 keeps its existing stage pipeline and adds bounded execution queues around it:

- CPU queue: transcription, translation, analysis, and other host work.
- GPU queue: a single serialized upscaling stream, preventing competing jobs from exhausting VRAM.
- I/O queue: source prefetching plus optional background upload and archive finalization.

`parallel_enabled` allows uploads from a completed job to overlap CPU/GPU work for the next job. Source prefetch reads only the first MiB, warming filesystem or mounted-drive caches without retaining videos in RAM. Job leases prevent multiple workers from scheduling the same video.

Persistent stage-cache keys contain the source fingerprint, stage name, and a stable hash of only that stage's relevant configuration. Whisper, translation, analysis/metadata, and preview results are therefore reused after restarts but invalidated when their input or relevant settings change.

Interrupted `in_progress` and `retrying` stages return to `pending` at startup. Completed stages remain completed. Incomplete encoding/upscaling outputs are discarded before retry, while resumable upload session files remain available to the Drive client.

Resource checks record CPU, RAM, GPU utilization, used VRAM, and disk capacity. Adaptive batching uses free VRAM with a conservative two-GiB-per-batch estimate and respects `max_batch_size`. Stale `.tmp`, `.part`, and generated upscaling intermediates are cleaned according to `cleanup_max_age_seconds`.

## Benchmarking and profiling

Every stage writes elapsed time, work units, rate, cache status, and a resource snapshot to SQLite `benchmark_history`. Frame-based stages report FPS, transcription reports media-seconds per wall-second, and uploads report bytes per second. Reports aggregate average total time, bottleneck, GPU utilization, encoder FPS, cache-hit rate, skipped work, and estimated saved time.

Run optional profiling with:

```bash
los80 profile --output-dir reports/performance
```

This produces Python `pstats`, a folded-stack file accepted by common flamegraph tooling, and a cumulative-time list of the 50 slowest functions. SQLite and stage calls appear in the same profile, while durable stage timings remain in `benchmark_history`.

## Benchmark results

The automated regression suite performs 200 durable benchmark inserts plus aggregation in under five seconds on the test host. This is a guardrail, not a media-throughput claim. Actual restoration speed is dominated by source resolution, Real-ESRGAN model, scale factor, subtitle model, encoder, storage, thermals, and whether frames are decoded in hardware.

Use each library's own `benchmark_history` as the authoritative result. A representative benchmark should include at least three clips totaling 20–30 minutes, with cold-cache and warm-cache runs reported separately.

## Recommended hardware

- Storage: local NVMe scratch space with at least twice the largest source size plus the configured reserve. Keep archives and reports on durable storage.
- Memory: 16 GiB minimum for serial processing; 32 GiB or more for concurrent transcription and uploads.
- GPU: NVIDIA hardware with at least 8 GiB VRAM for reliable tiled 1080p restoration. More VRAM primarily permits larger tiles and batches.
- CPU: 8 or more modern cores help decoding, subtitle generation, software encoding, and concurrent I/O.

## Expected throughput

The ranges below are planning estimates for a typical 1080p-to-4K Real-ESRGAN restoration with tiled processing and H.265 output. They are expressed as source-video minutes processed per wall-clock hour and must not be treated as guaranteed benchmarks.

| GPU | Expected source minutes/hour | Suggested starting batch | Operational note |
|---|---:|---:|---|
| T4 | 3–10 | 1 | Use 256-pixel tiles; avoid concurrent GPU work. |
| L4 | 10–25 | 2 | Good Colab balance; hardware encoding reduces CPU contention. |
| RTX 3060 | 7–18 | 1–2 | Prefer the 12 GiB model for larger tiles. |
| RTX 4070 | 15–35 | 2 | Strong single-job throughput with modest power use. |
| RTX 4090 | 30–70 | 3–4 | Storage and decode can become the bottleneck. |

Measure one representative title before scheduling a large library. LOS80's recorded FPS and estimated processing time provide a better forecast after the first completed job.

## Colab recommendations

- Put the SQLite database, reports, cache artifacts, and completed output on mounted Drive so a runtime disconnect does not lose state.
- Keep transient frame/intermediate work on `/content` for speed, with automatic cleanup enabled.
- Set `parallel_enabled: true`, `cpu_workers: 2`, and `io_workers: 2`; increasing these values usually adds contention on shared Colab storage.
- Use `analysis_quality_threshold` to avoid unnecessary GPU passes and enable animated GIF previews only when needed.
- After reconnecting, rerun the same command. Recovery resets only interrupted stages, cache keys reuse valid work, and Drive upload session files allow transfer continuation.
