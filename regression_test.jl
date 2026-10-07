using Test, TensorQEC

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
