using ConformalAmp
using StableRNGs: StableRNG
using Plots
using ProgressBars

rng = StableRNG(0)

α = 0.5
λ = 1.0
d = 50

problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)

X, w, y = ConformalAmp.sample_all(rng, problem, d; model = "laplace")
X_test  = ConformalAmp.sample_data_any_n(rng, d, 1000; model = "laplace")
y_test = ConformalAmp.sample_labels(rng, problem, X_test, w)

problem = ConformalAmp.Ridge(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = 1.0)

ŵ_erm  = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
ŵ_gamp = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(max_iter = 10, rtol = 1e-2))

y_pred = X_test * ŵ_erm
y_pred_gamp = X_test * ŵ_gamp

fcp = ConformalAmp.FullConformal(coverage = 0.9, δy_range = 0.05:0.05:5.0)


for i in ProgressBar(1:1)
    xtest = X_test[i, :]
    result_erm = ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, ConformalAmp.ERM())
end