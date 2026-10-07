using TensorQEC

tanner = CSSTannerGraph(SurfaceCode(3, 3))
n = TensorQEC.nq(tanner)
p = 0.03
problem = IndependentDepolarizingDecodingProblem(tanner, iid_error(p / 3, p / 3, p / 3, n))
pm = problem.pvec

# 1. get_probability only looks at qubit 1
w1 = (1, 0)          # X on qubit 1           -> true prob ~ 0.01 * 0.97^8
w2 = (6, 0)        # X on qubits 2 and 3    -> true prob ~ 0.01^2 * 0.97^7
println("length(cep[1]) = ", length(w1[1]))
println("get_probability(X1)     = ", TensorQEC.get_probability(pm, w1))
println("get_probability(X2 X3)  = ", TensorQEC.get_probability(pm, w2))
println("(expected X1 > X2X3: ", 0.01 * 0.97^8, " vs ", 0.01^2 * 0.97^7, ")")

# 2. a single X1 error is decoded into a logical error
lx, lz = logical_operator(tanner)
x = zeros(Mod2, n); x[1] = 1
err = CSSErrorPattern(x, zeros(Mod2, n))
res = decode(compile(TableDecoder(4), problem), syndrome_extraction(err, tanner))
println("X1 decoded as: x=", findall(isone, res.error_pattern.xerror), " z=", findall(isone, res.error_pattern.zerror))
println("logical error after correction: ", check_logical_error(err, res.error_pattern, lx, lz))
