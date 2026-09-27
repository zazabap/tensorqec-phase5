# Bug report (draft for upstream nzy1997/TensorQEC.jl)

**Title:** `TableDecoder` conflict resolution uses only the first qubit's probability (`length(Int) == 1`)

**Package:** TensorQEC.jl (verified against the source at `src/decoding/truthtable.jl`, present in the installed release)

## Summary

In `get_probability(pm::IndependentDepolarizingError, cep)` (truthtable.jl), the line

```julia
qubit_num = length(cep[1])
```

takes `length` of a **packed `Int`** (the x-error bitstring). In Julia,
`length(::Number) == 1`, so the "probability" used by `conflict_syndrome` only
considers the **first qubit** of each candidate error pattern. As a result,
`make_table` stores sub-optimal error patterns for many syndromes: any weight-≤4
explanation that leaves qubit 1 untouched (probability ≈ 1−p) beats any candidate
touching qubit 1 (probability ≈ p), regardless of true likelihood.

## Reproduction

```julia
using TensorQEC
tanner = CSSTannerGraph(SurfaceCode(3,3))
p = 0.03
problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p/3, p/3, p/3, 9))
decoder = compile(TableDecoder(4), problem)   # built-in, buggy conflict resolution
```

- SurfaceCode(3,3), p=0.03, 2000 samples: built-in FER ≈ **0.0335**
- Same table rebuilt with correct full-pattern conflict resolution: FER ≈ **0.0130**
- Exact TN decoder (`TNMMAP`) reference: FER ≈ **0.0130** (6000 samples: 0.0132 vs
  0.0130, z = 0.08 — statistically indistinguishable)

## Mechanism evidence (deterministic, not statistical)

Decoding each of the 27 weight-1 Pauli errors on SurfaceCode(3,3) with both tables:

| true error | built-in table stores | corrected table stores | logical error (built-in) |
|---|---|---|---|
| X1 | X{2,3} | X{1} | **yes** |
| Y1 | X{2,3} Z{2} | X{1} Z{1} | **yes** |
| Z1 | Z{2} | Z{1} | no (differs only by a stabilizer) |

The built-in table mis-decodes exactly {X1, Y1} of 27 weight-1 errors (2/27), the
corrected table 0/27. Every observed replacement is a qubit-1-clean candidate, exactly
as the one-qubit-probability mechanism predicts.

## Root cause & fix

```julia
# current (buggy)
qubit_num = length(cep[1])          # == 1, because cep[1] is a packed Int

# fix: the number of qubits must be known here, e.g. stored in the error model or
# passed to make_table (TruthTable already knows num_qubits)
```

Full adversarial verification (mechanism + z-test + source check) is published in
`adversarial_check.jl` and `results/adversarial_verdict.txt` of this repository.

## Impact

Affects any user treating `TableDecoder` as a ground-truth/exact baseline: FER is
inflated (2.4x at p=0.03 on a d=3 code in our test). `TNMMAP`/`BPOSD` paths are
unaffected.
