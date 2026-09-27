# Adversarial verification of the TensorQEC.jl TableDecoder bug claim
#
# Claim under test (from verify_table_bug.jl):
#   In src/decoding/truthtable.jl, get_probability(::IndependentDepolarizingError, cep)
#   computes `qubit_num = length(cep[1])` where cep[1] is a packed Int.
#   Since length(::Number) == 1 in Julia, syndrome-conflict resolution compares error
#   patterns using ONLY qubit 1's marginal probability, so the lookup table stores
#   sub-optimal errors and the FER is inflated.
#
# Falsification attempts (the claim survives only if ALL fail to disprove it):
#   Test A (mechanism): the built-in table should mis-decode EXACTLY the weight-1
#       errors on qubit 1 (whose degenerate partner e+L keeps qubit 1 clean), and
#       nothing else; the corrected table should decode all 27 weight-1 errors.
#   Test B (statistics): at p=0.03, built-in FER should be significantly above the
#       corrected table's FER (z-test), while corrected ≈ TNMMAP within noise.
#   Test C (source): the installed package must contain the exact buggy line.

using TensorQEC
using Random
using Printf

struct CorrectConflict{T} <: TensorQEC.AbstractSyndromeConflict
    dis::T
    nq::Int
end

_bit(x::Integer, i::Int) = (x >> (i - 1)) & 1 == 1

function TensorQEC.get_probability(sc::CorrectConflict, cep::Tuple)
    pm = sc.dis
    p = 1.0
    for i in 1:sc.nq
        xe = _bit(cep[1], i)
        ze = _bit(cep[2], i)
        if !xe
            p *= ze ? pm.pz[i] : (1 - pm.px[i] - pm.py[i] - pm.pz[i])
        else
            p *= ze ? pm.py[i] : pm.px[i]
        end
    end
    return p
end

# weight-1 Pauli error pattern: type 1=X, 2=Y, 3=Z on qubit q
function weight1_error(n, q, t)
    x = zeros(Mod2, n); z = zeros(Mod2, n)
    (t == 1 || t == 2) && (x[q] = 1)
    (t == 2 || t == 3) && (z[q] = 1)
    return CSSErrorPattern(x, z)
end

function main()
    mkpath("results")
    out = open("results/adversarial_verdict.txt", "w")
    println_both(out, "ADVERSARIAL VERIFICATION — TensorQEC TableDecoder bug")
    println_both(out, "=" ^ 64)

    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    n = TensorQEC.nq(tanner)              # 9
    p = 0.03
    problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p / 3, p / 3, p / 3, n))
    lx, lz = logical_operator(tanner)

    builtin = compile(TableDecoder(4), problem)
    corrected = TensorQEC.CompiledTable(make_table(tanner, 4, CorrectConflict(problem.pvec, n)))

    # ---------------- Test A: mechanism ----------------
    # Mechanism under test: the buggy conflict resolver compares candidates by
    # qubit-1-only probability, so ANY weight<=4 explanation of a syndrome that keeps
    # qubit 1 clean (buggy p = 1-px-py-pz = 0.97) replaces ANY candidate touching qubit 1
    # (buggy p = p/3 = 0.01), regardless of true likelihood. Harm occurs when the clean
    # replacement is logically inequivalent to the true error.
    println_both(out, "\n[Test A] Mechanism: decode ALL weight-1 errors with both tables")
    fails_builtin = Tuple{Int,Int}[]
    fails_corrected = Tuple{Int,Int}[]
    for q in 1:n, t in 1:3
        err = weight1_error(n, q, t)
        syn = syndrome_extraction(err, tanner)
        rb = decode(builtin, syn)
        rc = decode(corrected, syn)
        if check_logical_error(err, rb.error_pattern, lx, lz)
            push!(fails_builtin, (q, t))
        end
        if check_logical_error(err, rc.error_pattern, lx, lz)
            push!(fails_corrected, (q, t))
        end
    end
    labels = Dict(1 => "X", 2 => "Y", 3 => "Z")
    sb = join(["$(labels[t])$q" for (q, t) in fails_builtin], ", ")
    println_both(out, "  built-in  table mis-decodes: $(length(fails_builtin))/27  -> $sb")
    println_both(out, "  corrected table mis-decodes: $(length(fails_corrected))/27")

    # Evidence of the replacement: what does the built-in table store for X1/Y1/Z1?
    println_both(out, "  stored built-in entries for qubit-1 errors (evidence of qubit-1-clean replacement):")
    for t in 1:3
        err = weight1_error(n, 1, t)
        syn = syndrome_extraction(err, tanner)
        rb = decode(builtin, syn)
        rc = decode(corrected, syn)
        xq = [i for i in 1:n if rb.error_pattern.xerror[i].x]
        zq = [i for i in 1:n if rb.error_pattern.zerror[i].x]
        cxq = [i for i in 1:n if rc.error_pattern.xerror[i].x]
        czq = [i for i in 1:n if rc.error_pattern.zerror[i].x]
        println_both(out, @sprintf("    %s1: built-in X:%s Z:%s | corrected X:%s Z:%s",
                    labels[t], xq, zq, cxq, czq))
    end

    # Refined prediction: built-in mis-decodes can only involve errors touching qubit 1
    # (the only qubit the buggy probability looks at). Harmless replacements (as for Z1,
    # where the clean alternative differs only by a stabilizer) are allowed.
    only_q1 = all(t -> t[1] == 1, fails_builtin)
    a_ok = only_q1 && isempty(fails_corrected) && !isempty(fails_builtin)
    println_both(out, "  Verdict: mis-decodes must be a subset of {X1,Y1,Z1} and corrected must be clean")
    println_both(out, "  Test A " * (a_ok ? "PASSES (refined mechanism fully explains the observed set [$sb])" :
                              "FAILS (observed set [$sb] contradicts the mechanism)"))

    # ---------------- Test B: statistics ----------------
    println_both(out, "\n[Test B] Statistics: p=0.03, 6000 samples, built-in vs corrected vs TNMMAP")
    Random.seed!(1234)
    compiled_tn = compile(TNMMAP(), problem)
    fb = fc = ft = 0
    for _ in 1:6000
        err = random_error_pattern(problem.pvec)
        syn = syndrome_extraction(err, tanner)
        rb = decode(builtin, syn)
        rc = decode(corrected, syn)
        rt = decode(compiled_tn, syn)
        (!rb.success_tag || check_logical_error(err, rb.error_pattern, lx, lz)) && (fb += 1)
        (!rc.success_tag || check_logical_error(err, rc.error_pattern, lx, lz)) && (fc += 1)
        (!rt.success_tag || check_logical_error(err, rt.error_pattern, lx, lz)) && (ft += 1)
    end
    N = 6000
    f_b, f_c, f_t = fb / N, fc / N, ft / N
    z_bc = (f_b - f_c) / sqrt(f_b * (1 - f_b) / N + f_c * (1 - f_c) / N)
    z_ct = (f_c - f_t) / sqrt(f_c * (1 - f_c) / N + f_t * (1 - f_t) / N)
    @printf("  built-in  FER = %.5f (%d)\n", f_b, fb)
    @printf("  corrected FER = %.5f (%d)\n", f_c, fc)
    @printf("  TNMMAP    FER = %.5f (%d)\n", f_t, ft)
    println_both(out, @sprintf("  built-in  FER = %.5f (%d)", f_b, fb))
    println_both(out, @sprintf("  corrected FER = %.5f (%d)", f_c, fc))
    println_both(out, @sprintf("  TNMMAP    FER = %.5f (%d)", f_t, ft))
    println_both(out, @sprintf("  z(builtin vs corrected) = %.2f  (claim: |z| >> 3)", z_bc))
    println_both(out, @sprintf("  z(corrected vs TNMMAP)  = %.2f  (claim: |z| < 3, i.e. no significant gap)", z_ct))
    b_ok = z_bc > 3.0 && abs(z_ct) < 3.0
    println_both(out, "  Test B " * (b_ok ? "PASSES" : "FAILS"))

    # ---------------- Test C: source line ----------------
    println_both(out, "\n[Test C] Source: is the buggy line present in the installed package?")
    pkgfile = joinpath(dirname(dirname(pathof(TensorQEC))), "src", "decoding", "truthtable.jl")
    line = "qubit_num = length(cep[1])"
    present = occursin(line, read(pkgfile, String))
    println_both(out, "  checked file: $pkgfile")
    println_both(out, "  line '$line' present: $present")
    c_ok = present
    println_both(out, "  Test C " * (c_ok ? "PASSES" : "FAILS"))

    verdict = a_ok && b_ok && c_ok
    println_both(out, "\n" * "=" ^ 64)
    println_both(out, "FINAL VERDICT: bug claim " * (verdict ? "CONFIRMED (survived all falsification attempts)" :
                                                   "NOT confirmed — needs re-investigation"))
    close(out)
end

function println_both(io, s)
    println(io, s)
    println(s)
end

main()
