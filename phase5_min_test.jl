# Phase 5 minimum test — BP vs TN decoding on small quantum (QC-)LDPC codes
#
# What this does:
#   For each small CSS code, sweep the physical depolarizing rate p and estimate the
#   logical error rate (FER) of three decoders on the SAME sampled errors:
#     1. BPDecoder()     — belief propagation (with OSD post-processing) [Kasai-style, classical split]
#     2. TNMMAP()        — tensor-network marginal-MAP decoder (exact TN contraction)
#     3. TableDecoder(4) — bounded-weight lookup table (near-exact ground truth for small codes)
#
# Output: a summary table on stdout + CSV files in ./results/

using TensorQEC
using Random
using Printf

const RNG = Xoshiro(0x5eed5eed)

function fer_sweep(tanner, label; ps, nsamples)
    n = TensorQEC.nq(tanner)
    nstab = TensorQEC.ns(tanner)
    decoders = Dict{String,Any}(
        "BP(+OSD)"      => BPDecoder(),
        "TNMMAP"        => TNMMAP(),
        "Table(d=4)"    => TableDecoder(4),
    )
    lx, lz = logical_operator(tanner)
    @printf("\n=== %s: nq=%d, nstab=%d, k=%d ===\n", label, n, nstab, size(lx, 1))

    rows = String[]
    for p in ps
        problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p / 3, p / 3, p / 3, n))
        compiled = Dict{String,Any}()
        for (name, dec) in decoders
            t0 = time()
            compiled[name] = compile(dec, problem)
            @printf("  compile %-12s @ p=%.3f : %.1fs\n", name, p, time() - t0)
        end
        fails = Dict(name => 0 for name in keys(decoders))
        for _ in 1:nsamples
            err = random_error_pattern(problem.pvec)
            syn = syndrome_extraction(err, tanner)
            for name in keys(decoders)
                res = decode(compiled[name], syn)
                is_logical_error = !res.success_tag ||
                                   check_logical_error(err, res.error_pattern, lx, lz)
                is_logical_error && (fails[name] += 1)
            end
        end
        for name in keys(decoders)
            fer = fails[name] / nsamples
            push!(rows, join([label, string(p), name, string(fer), string(nsamples)], ","))
            @printf("  p=%.3f  %-12s FER = %.4f  (%d/%d)\n", p, name, fer, fails[name], nsamples)
        end
    end
    return rows
end

function main()
    mkpath("results")
    header = "code,p,decoder,fer,nsamples"
    allrows = [header]

    # 1) Surface code 3x3: [[13,1,3]] — sanity check / warm-up
    append!(allrows, fer_sweep(CSSTannerGraph(SurfaceCode(3, 3)), "surface33",
                               ps = [0.03, 0.06, 0.09, 0.12, 0.15], nsamples = 300))

    # 2) Toric code 2x3: a genuine (tiny) bivariate-bicycle code, [[12,2,2]]
    append!(allrows, fer_sweep(CSSTannerGraph(ToricCode(2, 3)), "toric23",
                               ps = [0.03, 0.06, 0.09, 0.12, 0.15], nsamples = 300))

    # 3) Toric code 3x3: bivariate-bicycle on Z_3 x Z_3, [[18,2,3]]
    append!(allrows, fer_sweep(CSSTannerGraph(ToricCode(3, 3)), "toric33",
                               ps = [0.03, 0.06, 0.09, 0.12, 0.15], nsamples = 200))

    open("results/phase5_min_test.csv", "w") do io
        for r in allrows
            println(io, r)
        end
    end
    println("\nSaved: results/phase5_min_test.csv")
end

main()
