"""
Compare the test error of ERM and AMP for Lasso
"""

using Plots
using StableRNGs: StableRNG
using Statistics

using ConformalAmp

"""
α = 0.5
λ_list = (10.0).^(-3:0.2:0)

d = 100
ntest = 1000

test_error_erm_list = []
test_error_amp_list = []
for λ in λ_list


    rng = StableRNG(0)

    problem = ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = xtest * w

    ŵ_erm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
    ŵ_amp = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(rtol = 1e-3, max_iter = 100))

    test_error_erm = mean(( ytest - xtest * ŵ_erm ).^2)
    test_error_amp = mean(( ytest - xtest * ŵ_amp ).^2)

    push!(test_error_erm_list, test_error_erm)
    push!(test_error_amp_list, test_error_amp)
end

plt = scatter(λ_list, test_error_erm_list, label="ERM (d = $d)", xscale=:log, xaxis="λ", yaxis="ε_gen")
scatter!(plt, λ_list, test_error_amp_list, label="AMP (d = $d)", xscale=:log)
"""

"""
# Plot the overlaps as a function of λ

rng = StableRNG(10)
d = 500
n = Integer(0.5 * 500)
α = n / d

function create_problem(λ)
    return ConformalAmp.Lasso(α = α, λ = λ, Δ = 1.0, Δ̂ = 1.0) 
end

true_problem = create_problem(1.0)

λ_list = (10.0).^(-3:0.2:0)

q_list = []
m_list = []

for λ in λ_list
    problem = create_problem(λ)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    ŵ = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(rtol = 1e-3, max_iter = 100))
    q = ŵ' * ŵ / d
    m = ŵ' * w / d
    push!(q_list, q)
    push!(m_list, m)
end

plt = scatter(λ_list, q_list, label="q", xscale=:log, xaxis="λ", yaxis="overlap")
scatter!(plt, λ_list, m_list, label="m", xscale=:log)
display(plt)
"""

