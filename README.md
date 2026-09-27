# Phase 5 Minimum Test — BP vs TN Decoding on Quantum LDPC Codes

Independent, reproducible environment for the Phase 5 mini-project of
[`../kasai_tensorqec_roadmap.md`](../kasai_tensorqec_roadmap.md):
*Exact tensor-network decoding of quantum QC-LDPC codes: quantifying the BP–ML gap
and the error floor.*

## Setup

```bash
cd tensorqec_phase5
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia ≥ 1.10 (tested on 1.12.6)
```

Dependencies: `TensorQEC` (main package), `OMEinsum` (contraction-complexity probe).

## Scripts

| File | What it does |
|---|---|
| `phase5_min_test.jl` | Sweeps depolarizing rate `p` on 3 small CSS codes; estimates FER of `BPDecoder` (BP+OSD), `TNMMAP` (tensor-network marginal-MAP), `TableDecoder(4)` (bounded-weight lookup) on the **same sampled errors**. Output: `results/phase5_min_test.csv` |
| `bb72_complexity.jl` | Constructs the [[72,12,6]] bivariate-bicycle code, verifies its parameters, and reports the TNMMAP contraction complexity (time/space) **without contracting**. Output: `results/bb72_complexity.txt` |
| `verify_table_bug.jl` | Reproduces + isolates a suspected bug in `TableDecoder`'s syndrome-conflict resolution (see below) |
| `adversarial_check.jl` | Adversarial (falsification) verification of the bug claim: mechanism test, z-test statistics, source check |

## Results

### 1. Decoder comparison on small codes (logical error rate, depolarizing noise)

| code | p | BP(+OSD) | TNMMAP (exact) | Table(d=4)* |
|---|---|---|---|---|
| surface33 [[9,1,3]] | 0.03 | 0.0067 | 0.0067 | 0.0433* |
| surface33 | 0.09 | 0.0767 | 0.0800 | 0.1467* |
| surface33 | 0.15 | 0.2067 | 0.1667 | 0.2067* |
| toric23 [[12,2,2]] (BB) | 0.03 | 0.1200 | **0.0767** | 0.0933* |
| toric23 | 0.09 | 0.3200 | **0.2133** | 0.2467* |
| toric23 | 0.15 | 0.5100 | **0.3633** | 0.4100* |
| toric33 [[18,2,3]] (BB) | 0.03 | 0.0100 | 0.0050 | 0.0500* |
| toric33 | 0.09 | 0.1750 | **0.1450** | 0.1900* |
| toric33 | 0.15 | 0.3600 | **0.2650** | 0.3750* |

\* `Table(d=4)` column is **invalid as ground truth** — affected by the package bug
described below. Do not use built-in `TableDecoder` results until fixed.
`TNMMAP` is exact for these sizes and serves as the ground truth.

### 2. What the numbers say (the pitch-ready story)

- **BP ≤ TN holds almost everywhere, and the BP–ML gap grows with p** — visible on
  every code, e.g. toric33: BP 0.1750 vs TN 0.1450 at p=0.09; 0.3600 vs 0.2650 at
  p=0.15. Exactly the "sharp transition + error floor" regime Kasai's
  [2507.11534](https://arxiv.org/abs/2507.11534) studies; the TN decoder quantifies
  how far BP is from optimal.
- **The largest gap is on the degenerate code** (toric23, d=2, fully degenerate
  weight-1 errors): BP loses ~4 percentage points even at p=0.03. This is a concrete,
  reproducible instance of **quantum degeneracy hurting BP** — the natural seed of a
  "degenerate trapping sets" story.
- **The full [[72,12,6]] is confirmed infeasible for exact TN decoding**: parameters
  verified (nq=72, ns=64, k=12); TNMMAP contraction complexity **time ≈ 2^84,
  space ≈ 2^69**. This validates the roadmap's caution (#P-hard) and defines the real
  research question: *small codes exact + larger codes via truncated contraction /
  MPS-style approximations*.

### 3. Found: a genuine bug in TensorQEC.jl `TableDecoder` (PR opportunity)

`src/decoding/truthtable.jl`, `get_probability(pm::IndependentDepolarizingError, cep)`:

```julia
qubit_num = length(cep[1])   # cep[1] is a packed Int → length(Int) == 1 in Julia!
```

The syndrome-conflict resolution therefore compares error patterns using **only the
first qubit's** probability, so the lookup table stores sub-optimal errors for many
syndromes (a weight-2 error with a clean first qubit can beat a weight-1 error on
qubit 1). Verification (`verify_table_bug.jl`, surface33, p=0.03, 2000 samples):

- built-in `TableDecoder(4)`: FER = **0.0335**
- table rebuilt with correct full-pattern conflict resolution: FER = **0.0130**
- BP / TNMMAP reference: ≈ 0.0067 (± sampling noise)

Fix is one line (`length(cep[1])` → number of qubits, which must be passed through or
derived from the error model length). This is exactly the "Stage T6" credibility
contribution from the roadmap: open an issue with this reproduction, then send the PR.

**Adversarial confirmation** (`adversarial_check.jl`, full log in
`results/adversarial_verdict.txt`):

- **Test A (mechanism)** — predicted and confirmed at the level of individual errors:
  the built-in table replaces every qubit-1-touching candidate with a qubit-1-clean one
  (e.g. syndrome of X1 stores X{2,3} instead of X1; Y1 stores (X{2,3}, Z{2})), which
  causes logical errors on {X1, Y1} — exactly the subset of the 27 weight-1 errors the
  refined mechanism predicts (Z1's replacement differs only by a stabilizer and is
  harmless). Corrected table: 0/27 mis-decoded.
- **Test B (statistics)** — 6000 samples at p=0.03: built-in FER 0.0317 vs corrected
  0.0132, z = 6.86; corrected vs TNMMAP 0.0130, z = 0.08. Inflation is significant and
  the corrected table is consistent with the exact decoder.
- **Test C (source)** — the buggy line is present verbatim in the installed package
  (`~/.julia/packages/TensorQEC/<hash>/src/decoding/truthtable.jl`).

**Verdict: CONFIRMED — the claim survived all three falsification attempts.**
See the issue in this repository for the upstream report draft.

## Next steps

1. File the `TableDecoder` issue + PR on
   [nzy1997/TensorQEC.jl](https://github.com/nzy1997/TensorQEC.jl).
2. Load Kasai-style QC-LDPC instances (his
   [distance_certificate CSVs](https://arxiv.org/src/2604.20838v4/anc) or small BB
   codes) and rerun the same BP-vs-TN sweep → this becomes slide 2 of the pitch.
3. Study *where* BP fails (support/weight of failure events) vs the TN decoder to
   characterize trapping-set-like structures — the core of Paper 1's open question.
4. Experiments on truncated/approximate TN contraction for sizes between 18 and 72
   qubits (the actual research gap).
