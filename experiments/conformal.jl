"""
Script to compare the intervals that are returned by the LOO _vs_ conformal prediction 
"""

using ConformalAmp
using StableRNGs: StableRNG

# define the problem and sample the data
d   = 500
α = 0.5
λ = 1e-6
coverage = 0.9

problem = ConformalAmp.Ridge(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)

rng = StableRNG(0)
(; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

xtest = ConformalAmp.sample_data_any_n(rng, d, 1)[1, :]

jplus_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.JacknifePlus(coverage = coverage), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))
fcp_confidence_set = ConformalAmp.get_confidence_interval(problem, X, y, xtest, ConformalAmp.FullConformal(coverage = coverage, δy_range = 0:0.1:2.0), ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4))

println("J+  :",  minimum(jplus_confidence_set), " ", maximum(jplus_confidence_set))
println("FCP : ", minimum(fcp_confidence_set), " ", maximum(fcp_confidence_set))