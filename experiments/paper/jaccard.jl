"""
Plot to compute the jaccard index between exact FCP and AMPTaylor, as well as split conformal prediction
"""

using ProgressBars
using StableRNGs: StableRNG
using Statistics

using ConformalAmp

function jaccard_index(a1::Real, b1::Real, a2::Real, b2::Real)
    intersection = max(0, min(b1, b2) - max(a1, a2))
    union = max(b1, b2) - min(a1, a2) # not correct if the intervals are disjoint but does not matter because the intersection is 0
    return intersection / union
end

function compute_jaccard_amp_scp(problem::ConformalAmp.Problem; d::Integer = 100, ntest::Integer = 10, seed::Integer = 0)
    """
    Compare the confidence interval given by ERM() so by refitting everything and GAMPTaylor for a
    single test point at a fixed dimension
    """
    rng = StableRNG(seed)

    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest_array = ConformalAmp.sample_data_any_n(rng, d, ntest)

    algo = ConformalAmp.FullConformal(δy_range = 0.0:0.05:5.0, coverage = 0.9)
    scp_algo = ConformalAmp.SplitConformal(coverage = 0.9)

    jaccard_list_erm_amptaylor = []
    jaccard_list_erm_scp = []

    for i in 1:ntest
        xtest = xtest_array[i, :]
        ci_erm = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, ConformalAmp.ERM())
        ci_amptaylor = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo,     
                                ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4))

        ci_scp = ConformalAmp.get_confidence_interval(problem, X, y, xtest, scp_algo, ConformalAmp.ERM())

        jaccard_amp = jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_amptaylor), maximum(ci_amptaylor)) 
        jaccard_scp = jaccard_index(minimum(ci_erm), maximum(ci_erm), minimum(ci_scp), maximum(ci_scp)) 
        push!(jaccard_list_erm_amptaylor, jaccard_amp)
        push!(jaccard_list_erm_scp, jaccard_scp)
    end
    return jaccard_list_erm_amptaylor, jaccard_list_erm_scp
end

α = 2.0
d = 50

problems = [
    ConformalAmp.Lasso(α = α, λ = 0.01, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Lasso(α = α, λ = 0.1, Δ = 1.0, Δ̂ = 1.0),
    ConformalAmp.Lasso(α = α, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
]

for problem in problems
    jaccard_list_erm_amptaylor, jaccard_list_erm_scp = compute_jaccard_amp_scp(problem; d=d, ntest=50, seed=0)

    println("Mean for $problem : $(mean(jaccard_list_erm_amptaylor)), $(mean(jaccard_list_erm_scp))")
    # print standard deviation
    println("Std for $λ : $(std(jaccard_list_erm_amptaylor)), $(std(jaccard_list_erm_scp))")
end