using ConformalAmp
using ProgressBars
using StableRNGs: StableRNG
rng = StableRNG(0)

problems = [
    ConformalAmp.Pinball(α = 0.5, λ = 0.01, Δ = 1.0, q = 0.5),
    ConformalAmp.Pinball(α = 0.5, λ = 0.1, Δ = 1.0, q = 0.5),
    ConformalAmp.Pinball(α = 0.5, λ = 1.0, Δ = 1.0, q = 0.5),
]

gamp = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)
fcp = ConformalAmp.FullConformal(coverage = 0.9, δy_range = 0.0:0.05:10.0)
d = 250
n_test = 100

for problem in problems
    ci_mat = zeros(n_test, 2)
    
    count = 0

    for i in ProgressBar(1:n_test)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d; model = "gaussian")
        xtest = ConformalAmp.sample_data_any_n(rng, d, 1; model = "gaussian")
        ytest = ConformalAmp.sample_labels(rng, problem, xtest, w)
        ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest[1, :], fcp, gamp)
        if minimum(ci) <= ytest[1] <= maximum(ci)
            count += 1
        end
    end
    
    coverage = count / n_test
    # compute the coverage

    println("For $problem, coverage is $coverage")
end