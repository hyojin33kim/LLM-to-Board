# LLM Tile Design Lab
Private browser-based Stories260K partition explorer. Static assets in dist/.

## Verified
- 60 configurations match Python v0.3 partitioner: feasibility, stage cycles, throughput and selected cuts.
- Browser QA: default 3+2 recommendation, SRAM infeasibility, save/restore and alternate candidate selection.
- Browser module syntax checks pass.

Run `node tests/parity.mjs` from this directory.

The JavaScript engine ports the Python contiguous partitioner. All time values are assumed model calculations, not hardware measurements. Profile source: hyojin33kim/CV32E40P_Stories260K commit 661292a84ec02d5fda4ec5f61ea03f3a8e4a23cf. Models other than Stories260K, RTL generation and board execution are not implemented.

JSON export/import handles exposed UI settings; advanced non-UI parameters remain fixed to config.json. Browser save is localStorage on that browser, not account synchronization.

## v0.4 — executable-plan contract (declarative only)
`dist/planner.mjs` generates a serial FP32 decode-step plan for the selected contiguous partition, for one request only. It emits real buffer offsets and lifetimes, DDR checkpoint source offsets, <=4 KiB weight DMA blocks, row-complete GEMV chunks, QK/AV strided KV views, state commits, and inter-Tile copy descriptors. No operations execute numerically yet.

- Web input defaults to one request; partition exploration still supports multiple requests.
- Plan is token ID 1 at position context-1. Nonzero position requires previously computed KV prefix. These are declared preconditions, not preconditions we have executed or verified on a device.
- Weight source is the pinned Stories260K checkpoint SHA256 b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696.
- Classifier shares embedding in the checkpoint; separate tiles have separate table copies.
- Persistent KV/debug and scratch intervals are aligned to 64 bytes. Scratch first-fit allocation reuses addresses only after the previous lifetime ends. No reuse within the same descriptor.
- `DMA_LOAD` is an abstract transfer contract, not a physical AXI transaction. `GEMV_ROWS` completes whole dot products; no partial-K accumulator is required.
- `wait_for` enforces a strict serial completion chain. A descriptor completes only after all writes commit. Link completion similarly means destination data is available.
- FFN intermediate tensors never leave SRAM. This is locality, not proof of a fused arithmetic datapath or reduction equivalence.
- Input/output tensor regions encode offset, row count, row bytes, row stride. SRAM/DDR addresses are tile-local relative addresses, not final SoC mappings.
- Validator checks alignment, region bounds, tile ownership, declared initialization, ordering, recomputed buffer lifetimes, simultaneous address overlap and total capacity. It does not establish numerical correctness or detect hardware faults.
- Capacity uses actual lifetime allocation plus configured debug storage and 10% reserve; CPU RAM/ROM, IO, queues beyond weight staging, and future firmware space are not included. The old analytical screen uses a conservative fixed scratch/DMA reservation, so the two memory views differ.
- The upper timing estimate is still the v0.3 op-level model; no descriptor timing/launch/port scheduling has been calibrated.

Validation: `node tests/planner.mjs` checks 48 partitions across context1/128/512; all 188 profile graph nodes covered; 8 failure/negative checks include missing dependencies, out-of-bounds access, uninitialized weights, live aliasing, incorrect lifetime, ownership, capacity, and multiple requests. `node tests/parity.mjs` still matches the Python baseline on 60 cases.

The default 3+2 plan has 709 descriptors and 1,040,384 weight DMA bytes/token. T0 SRAM416512/DDR676352 bytes; T1 SRAM285440/DDR494848 bytes. These are plan allocations, not measured silicon area or RAM macro requirements.

Next: a functional descriptor interpreter and numerical Golden comparison, then cycle calibration and RTL generation. The UI does not claim these are implemented.

## v0.5 — functional descriptor execution
The browser now executes FP32 descriptors on tile-local SRAM/DDR arrays using the allocated addresses, including DMA, link copies, KV commit, RMSNorm, RoPE, GEMV, QK/Softmax/AV and SwiGLU. Trace START/END records retain `descriptor_id`, `tile`, and the descriptor's `graph_nodes`. Graph node IDs are stable semantic IDs; descriptor IDs are plan-local (one graph node can lower to multiple descriptors).

Run the web UI locally with `python -m http.server 8000 --directory dist`, then open http://localhost:8000. Select a partition, generate the plan, and run Golden comparison. Web Workers and SHA256 checks require localhost or HTTPS. No build dependencies are required.

Reproduce verification from the repository root (Python 3, Node 18+, C compiler, little-endian host):
```sh
python validation/rebuild.py
node tests/parity.mjs
node tests/planner.mjs
node tests/executor.mjs
```
- 48 numerical cases: all 16 contiguous partitions × contexts 1/128/512.
- 512 consecutive steps with interpreter-produced KV and split 3+2. Each step remaps/copies retained state into its generated plan; this is not a persistent hardware queue or fixed physical-memory test.
- Inputs are token ID 1 repeated; this does not validate tokenizer, text generation quality, arbitrary prompts, or other models.
- Browser tests use C-derived KV prefix for a single step. The continuous Node test never injects Golden KV.
- Comparison is elementwise `abs(actual-ref) <= 1e-5 + 1e-4*abs(ref)`, rejects nonfinite values. JS Math versus C libm is not bit-exact. This is a numerical test tolerance, not an agreed quantization quality budget.
- Layer-0 output corruption is detected, with first failing layer checkpoint identified. It does not prove the exact failing operation.
- Complete report: `dist/fixtures/validation-report.json`. C fixture provenance/hash: `dist/fixtures/manifest.json`.
- No RTL/NPU/FPGA execution or cycle/PPA calibration is claimed. Existing v0.3 time estimates remain assumptions.

### Reference provenance
`validation/run.c` and `dist/fixtures/weights.bin` are copied unchanged from the user's reference repository at commit `661292a84ec02d5fda4ec5f61ea03f3a8e4a23cf`:
https://github.com/hyojin33kim/CV32E40P_Stories260K
`golden_run.c` is generated by inserting only a layer-output observer into that source. The existing reference HW is not included or adopted as the new Tile architecture. Upstream code/model rights remain with their respective authors; no new license is asserted over reference assets.

GitHub is the engineering source destination; Sites hosts a separately synchronized private deployment. A GitHub push does not automatically redeploy Sites.
