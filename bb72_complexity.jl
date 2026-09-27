# Feasibility probe: exact TN decoding of the [[72,12,6]] bivariate-bicycle code
#
# Constructs the IBM-style [[72,12,6]] code (l = m = 6,
# A = x^3 + y + y^2, B = y^3 + x + x^2) and reports the time/space complexity of
# the TNMMAP (tensor-network marginal-MAP) decoder contraction — WITHOUT running
# the full contraction, which is expected to be infeasible exactly (see roadmap
# caution: exact ML decoding is #P-hard).
#
# If contraction turns out to be feasible (unlikely), we also run 5 samples as a
# bonus. Complexity output goes to results/bb72_complexity.txt.

using TensorQEC
using OMEinsum
using Printf

function main()
    mkpath("results")
    io = open("results/bb72_complexity.txt", "w")

    # BivariateBicycleCode(m, n, B-shifts, A-shifts); A = x^3 + y + y^2, B = y^3 + x + x^2
    bb = BivariateBicycleCode(6, 6, ((0, 3), (1, 0), (2, 0)), ((3, 0), (0, 1), (0, 2)))
    tanner = CSSTannerGraph(bb)
    n = TensorQEC.nq(tanner)
    lx, lz = logical_operator(tanner)
    msg = @sprintf("[[72,12,6]] check: nq=%d, ns=%d, k=%d\n", n, TensorQEC.ns(tanner), size(lx, 1))
    print(msg); println(io, msg)

    p = 0.05
    problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p / 3, p / 3, p / 3, n))

    t0 = time()
    ct = compile(TNMMAP(), problem)
    tcompile = time() - t0
    msg = @sprintf("TNMMAP compile time: %.1fs\n", tcompile)
    print(msg); println(io, msg)

    tc, sc = OMEinsum.contraction_complexity(ct)
    msg = @sprintf("TNMMAP contraction complexity: time ≈ 2^%.2f, space ≈ 2^%.2f\n", tc, sc)
    print(msg); println(io, msg)

    # Bonus: only if space complexity is small enough to risk one contraction
    if sc < 26.0
        t0 = time()
        err = random_error_pattern(problem.pvec)
        syn = syndrome_extraction(err, tanner)
        res = decode(ct, syn)
        le = check_logical_error(err, res.error_pattern, lx, lz)
        msg = @sprintf("BONUS single decode: %.1fs, logical error = %s\n", time() - t0, le)
        print(msg); println(io, msg)
    else
        msg = "Skipping actual contraction: infeasible exact memory/space (as expected).\n"
        print(msg); println(io, msg)
    end
    close(io)
end

main()
