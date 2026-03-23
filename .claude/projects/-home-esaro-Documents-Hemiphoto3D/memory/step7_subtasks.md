---
name: Step 7 Sub-tasks
description: PatchMatch MVS (Step 7) broken into 4 sub-tasks with kernel boundaries and status
type: project
---

## Step 7 — CUDA Dense Multi-View Stereo (PatchMatch MVS) Sub-tasks

### Sub-task 7A: Scaffolding, Source View Selection & Init Kernel ✅
- `patchmatch_mvs.h` with full API (`runPatchMatchMVS`, `selectSourceViews`, `SourceViewInfo`, `MVSViewParams`, PFM I/O)
- `patchmatch_mvs.cu` skeleton with memory management (texture objects, `DeviceBuffer` allocations)
- **Source view selection** (CPU): rank views by shared SfM points × baseline angle score, select top N_src
- **Kernel 1: `init_depth_normal_kernel`** — Philox cuRAND, random depth in [depth_min, depth_max], random normals facing camera
- Depth_min/depth_max from SfM sparse points (5th/95th percentile)
- `test_patchmatch.cpp`: 6 tests (source view selection, baselines, depth range, init depth, init normals, PFM round-trip)
- **Status: DONE** — 38/38 tests pass

### Sub-task 7B: NCC Cost Kernel ✅
- **Kernel 2: `compute_ncc_cost_kernel`** — plane-induced homography warping + NCC
- Precomputed `GPUSourceView` struct: A = K_src * R_rel * K_ref_inv, b = K_src * t_rel (host-side, uploaded)
- Efficient per-patch-pixel warp: q = A*p, factor = dot(nK, p)/n_dot_P, warped = q + b*factor
- Shared memory tile (26×26) for reference image patch reuse across source views
- NCC with running sums (count, sum_ref, sum_src, sum_ref2, sum_src2, sum_rs)
- Top-K=3 partial sort for multi-view cost aggregation (robust to occlusion)
- Texture objects created upfront for all registered images; source view params precomputed per ref view
- 3 new tests: cost values in range, cost variation, cost at SfM points < worst case
- **Status: DONE** — 41/41 tests pass

### Sub-task 7C: Propagation & Refinement Kernel ✅
- **`computePixelCost` device function** — extracted NCC cost computation for reuse by propagation kernel
- **Kernel 3: `propagate_and_refine_kernel`** — red-black checkerboard, spatial propagation (4 neighbors), random refinement
- Perturbation range shrinks exponentially: depth_range × 0.5^iter, normal_range × 0.5^iter
- Source view data uploaded once per ref view, reused across all iterations (no per-launch allocation)
- Wired into main PatchMatch iteration loop: init → NCC cost → N iterations × 2 checkerboard passes
- 1 new test: PropagationReducesCost (verifies mean cost decreases from 0→3 iterations)
- **Status: DONE** — 42/42 tests pass

### Sub-task 7D: Geometric Filter, Orchestration & End-to-End Testing ✅
- **`GeomFilterSrcView` struct** + **Kernel 4: `geometric_consistency_filter_kernel`** — cross-view depth consistency
- Backprojects reference pixel to 3D (world frame), reprojects into source views, checks depth agreement
- Normal check disabled (set to 180°) — normals converge too slowly with random-init PatchMatch
- Filter tolerances: 12% relative depth, min 2 consistent source views
- Only runs when `mvs_iterations > 0` (init-only depth maps are random noise)
- Packed source depth/normal upload per reference view; NCC confidence preserved for retained pixels
- Complete `runPatchMatchMVS` orchestration: source view selection → textures → init → NCC cost → propagation loop → filter
- 1 new combined end-to-end test: `EndToEndFilterAndAccuracy` (8 iterations, 36 views)
  - Sphere pixel retention ≥ 50%, background removal verified, mean depth error < 15%, performance < 5 min
- **Status: DONE** — 43/43 tests pass, MVS runs in ~73 sec for 36 views at 1280×960
