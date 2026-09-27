# Verify suspected bug in TensorQEC.jl TableDecoder conflict resolution
#
# Suspected bug (src/decoding/truthtable.jl, get_probability for
# IndependentDepolarizingError): `qubit_num = length(cep[1])` where cep[1] is a packed
# Int. In Julia, length(::Number) == 1, so the "probability" used to resolve syndrome
# conflicts only considers the FIRST qubit of each error pattern. The lookup table
# therefore stores sub-optimal errors for many syndromes (e.g. a weight-2 error with a
# clean first qubit beats a weight-1 error on qubit 1), inflating the logical error rate.
#
# This script rebuilds the table with a correct full-pattern conflict resolver and
# compares FER with the built-in TableDecoder on SurfaceCode(3,3) at p = 0.03.

using TensorQEC
using Random
using Printf

# Custom conflict resolver: compare full-pattern likelihood over ALL qubits
struct CorrectConflict{T} <: TensorQEC.AbstractSyndromeConflict
    dis::T     # IndependentDepolarizingError
    nq::Int    # number of qubits
end

function _bit(x::Integer, i::Int)
    return (x >> (i - 1)) & 1 == 1
end

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

function fer(decoder_compiled, tanner, problem, lx, lz, nsamples)
    fails = 0
    for _ in 1:nsamples
        err = random_error_pattern(problem.pvec)
        syn = syndrome_extraction(err, tanner)
        res = decode(decoder_compiled, syn)
        (!res.success_tag || check_logical_error(err, res.error_pattern, lx, lz)) && (fails += 1)
    end
    return fails / nsamples
end

function main()
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    n = TensorQEC.nq(tanner)
    p = 0.03
    problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p / 3, p / 3, p / 3, n))
    lx, lz = logical_operator(tanner)
    nsamples = 2000

    # Built-in TableDecoder(d=4) — uses DistributionError (suspected buggy)
    builtin = compile(TableDecoder(4), problem)

    # Rebuilt table — correct full-pattern conflict resolution
    tbl = make_table(tanner, 4, CorrectConflict(problem.pvec, n))
    corrected = TensorQEC.CompiledTable(tbl)

    t0 = time()
    fer_builtin = fer(builtin, tanner, problem, lx, lz, nsamples)
    t1 = time()
    fer_corrected = fer(corrected, tanner, problem, lx, lz, nsamples)
    t2 = time()

    @printf("surface33 p=%.3f, n=%d samples\n", p, nsamples)
    @printf("  TableDecoder(4) built-in : FER = %.4f  (%.1fs)\n", fer_builtin, t1 - t0)
    @printf("  Table rebuilt (correct)  : FER = %.4f  (%.1fs)\n", fer_corrected, t2 - t1)
    println("\nFor reference, BP(+OSD) and TNMMAP both gave FER ≈ 0.0067 at this point")
    println("(see results/phase5_min_test.csv). Correct table should match them.")
end

main()
