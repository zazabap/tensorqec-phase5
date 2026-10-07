# TensorQEC.jl `TableDecoder` bug: write-up and upstream issue

Reproduced 2026-10-02 against TensorQEC.jl v2.2.1 (the version pinned in this repo's
`Manifest.toml`) on Julia 1.12.6. The buggy line is also on upstream `main`
(`src/decoding/truthtable.jl:182`, last changed in commit `a8ef4b1`). No upstream issue
mentions it as of today.

Part 1 is the issue text, ready to paste into
<https://github.com/nzy1997/TensorQEC.jl/issues/new>. Part 2 is the full write-up for
our own records.

---

## Part 1 — Issue (to post upstream)

**Title:** `TableDecoder`: `get_probability` only looks at qubit 1 (`length(cep[1]) == 1`)

### Summary

When `make_table` finds two error patterns with the same syndrome, it keeps the more
likely one using `get_probability`. For `IndependentDepolarizingError` the qubit loop is

```julia
qubit_num = length(cep[1])
for i in 1:qubit_num
```

but `cep[1]` is the X part of the error packed into one integer, and `length` of a
number is 1 in Julia. So only qubit 1 is ever looked at. Any candidate that leaves
qubit 1 alone (probability `1 - px - py - pz`) beats any candidate that touches it
(probability `px`, `py` or `pz`), whatever the weight. The table then stores the wrong
correction for some syndromes, and `TableDecoder` can turn a weight-1 error into a
logical error.

### Minimal reproduction

```julia
using TensorQEC

tanner = CSSTannerGraph(SurfaceCode(3, 3))
em = iid_error(0.01, 0.01, 0.01, 9)

TensorQEC.get_probability(em, (1, 0))  # X1:    returns 0.01, expected 0.01 * 0.97^8 ≈ 7.8e-3
TensorQEC.get_probability(em, (6, 0))  # X2 X3: returns 0.97, expected 0.01^2 * 0.97^7 ≈ 8.1e-5

problem = IndependentDepolarizingDecodingProblem(tanner, em)
ct = compile(TableDecoder(4), problem)
lx, lz = logical_operator(tanner)
x = zeros(Mod2, 9); x[1] = 1
ep = CSSErrorPattern(x, zeros(Mod2, 9))
res = decode(ct, syndrome_extraction(ep, tanner))
findall(isone, res.error_pattern.xerror)             # [2, 3] instead of [1]
check_logical_error(ep, res.error_pattern, lx, lz)  # true
```

### Effect

On `SurfaceCode(3, 3)` at depolarizing rate p = 0.03:

- Of the 27 weight-1 Pauli errors, `TableDecoder(4)` turns 2 (X1 and Y1) into logical
  errors. With the fix below, none.
- Logical error rate over 6000 sampled errors: 3.17% with the current table, 1.32%
  with the fix, 1.30% with `TNMMAP` (exact). The fixed table agrees with the exact
  decoder (z = 0.08); the current one does not (z = 6.86).

`BPDecoder` and `TNMMAP` are not affected. `UniformError` is not affected (it returns
0 for every pattern). The existing `DistributionError` tests in
`test/decoding/truthtable.jl` ("DepolarizingDistribution" and "compile") only check
that the stored or decoded error reproduces the syndrome, never whether it causes a
logical error, which is probably why this was not caught. Both pass with and without
the fix.

### Suggested fix

The error model already has one entry per qubit, so the count can come from it:

```diff
 function get_probability(pm::IndependentDepolarizingError, cep::Tuple{INT, INT}) where {INT}
 	p = 1.0
-	qubit_num = length(cep[1])
+	qubit_num = length(pm.px)
 	for i in 1:qubit_num
```

With this change, the reproduction above decodes X1 as X1 and all 27 weight-1 errors
correctly. I can send a PR with this change and the regression test below.

### Regression test

Fails on the current code (4 failures), passes with the fix (29/29):

```julia
@testset "table conflict resolution uses all qubits" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = iid_error(0.01, 0.01, 0.01, 9)
    # X1 (weight 1) must be more likely than X2X3 (weight 2)
    @test TensorQEC.get_probability(em, (1, 0)) ≈ 0.01 * 0.97^8
    @test TensorQEC.get_probability(em, (1, 0)) > TensorQEC.get_probability(em, (6, 0))
    # every weight-1 Pauli error is decoded without a logical error
    problem = IndependentDepolarizingDecodingProblem(tanner, em)
    ct = compile(TableDecoder(4), problem)
    lx, lz = logical_operator(tanner)
    for q in 1:9, (xb, zb) in ((1, 0), (1, 1), (0, 1))
        x = zeros(Mod2, 9); z = zeros(Mod2, 9)
        x[q] = xb; z[q] = zb
        ep = CSSErrorPattern(x, z)
        res = decode(ct, syndrome_extraction(ep, tanner))
        @test !check_logical_error(ep, res.error_pattern, lx, lz)
    end
end
```

Environment: TensorQEC v2.2.1, Julia 1.12.6, Linux x86_64.

---

## Part 2 — Write-up (our records)

### Where it lives

`src/decoding/truthtable.jl`:

```julia
get_probability(de::DistributionError, cep::Tuple) = get_probability(de.dis, cep)
function get_probability(pm::IndependentDepolarizingError, cep::Tuple{INT, INT}) where {INT}
	p = 1.0
	qubit_num = length(cep[1])          # <- always 1
	for i in 1:qubit_num
		if iszero(readbit(cep[1], i))
			p = p * (iszero(readbit(cep[2], i)) ? (1-pm.px[i]-pm.py[i]-pm.pz[i]) : pm.pz[i])
		else
			p = p * (iszero(readbit(cep[2], i)) ? pm.px[i] : pm.py[i])
		end
	end
	return p
end
```

`compile(TableDecoder(d), problem)` builds the table with `make_table(tanner, d,
DistributionError(problem.pvec))`. `make_table` enumerates every error of weight up to
`d` and, when two errors share a syndrome, keeps the one with the larger
`get_probability`. A packed error is a `Tuple` of two integers (`Int` or
`BitBasis.LongLongUInt`), and `length(::Number) == 1`, so the loop runs once.

### Why it does damage

With the loop stuck at qubit 1, the "probability" of a candidate is just the
probability of whatever is on qubit 1. Any candidate that leaves qubit 1 alone scores
0.97 (at p = 0.03), any candidate that touches it scores 0.01. For a syndrome caused by
an error on qubit 1, the table therefore prefers some other explanation that avoids
qubit 1, even if it has higher weight.

That is harmless when the replacement differs from the true error by a stabilizer,
and a logical error otherwise. On the d = 3 surface code, what the current table
stores for the three single-qubit errors on qubit 1 is:

| true error | current table stores | fixed table stores | logical error (current) |
|---|---|---|---|
| X1 | X on {2, 3} | X1 | yes |
| Y1 | X on {2, 3}, Z on {2} | Y1 | yes |
| Z1 | Z2 | Z1 | no (Z1 Z2 is a stabilizer) |

Every replacement avoids qubit 1, as the mechanism predicts. Errors on other qubits
are unaffected only as long as some qubit-1-clean candidate is not competing for the
same syndrome; on larger codes or other qubit orderings, more syndromes can be hit.

### Why the fix is just `length(pm.px)`

The earlier draft (`docs/issue_tensorqec_tabledecoder_bug.md`) said the qubit count
"must be passed through or derived from the error model length". It doesn't need
passing through: `IndependentDepolarizingError` stores `px`, `py`, `pz` as vectors with
one entry per qubit, so `length(pm.px)` is the number of qubits. This also works for
`LongLongUInt`, since `readbit` already handles that type.

### What was run

All in this directory with the pinned `Manifest.toml`:

| script | result |
|---|---|
| `minimal_repro.jl` (new) | `get_probability(X1) = 0.01`, `get_probability(X2X3) = 0.97`; X1 decoded as X on {2, 3}, logical error `true` |
| `adversarial_check.jl` | reproduced `results/adversarial_verdict.txt` exactly: 2/27 vs 0/27; 3.17% vs 1.32% vs 1.30% (TNMMAP); z = 6.86 and 0.08 |
| `verify_table_bug.jl` | 2.50% current vs 1.35% fixed (2000 samples; no seed, so this differs from the 3.35% / 1.30% in the README) |
| `regression_test.jl` (new) | current code: 25 pass / 4 fail; with the fix applied as a method override: 29 / 29 pass |
| upstream `test/decoding/truthtable.jl` (2026-10-07) | all 6 testsets pass both without and with the fix (as a method override), so the fix breaks no existing test |

### Corrections to this repo's README

1. The README gives "BP / TNMMAP reference ≈ 0.0067" at p = 0.03 for surface33. That
   number comes from the small sweep in `results/phase5_min_test.csv`. With 6000
   samples, TNMMAP gives 1.30%, which matches the fixed table. The 0.67% is sampling
   noise and should not be quoted.
2. The fix is a one-line change using `length(pm.px)`, not a change to the
   `make_table` signature (see above).
3. `verify_table_bug.jl` does not set a random seed, so its numbers change on every
   run. `adversarial_check.jl` is the reproducible one.

### Next steps

1. Done 2026-10-07: posted as
   [nzy1997/TensorQEC.jl#144](https://github.com/nzy1997/TensorQEC.jl/issues/144).
2. Done 2026-10-07: fix and regression test in
   [nzy1997/TensorQEC.jl#145](https://github.com/nzy1997/TensorQEC.jl/pull/145),
   from `zazabap/TensorQEC.jl` branch `fix-table-probability`.
3. After it is merged, rerun the `Table(d=4)` column in `phase5_min_test.jl` and drop
   the "invalid" footnote from the README.
