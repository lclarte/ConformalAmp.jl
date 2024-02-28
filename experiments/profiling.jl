using ConformalAmp

d = 1000
λ = 0.1
model = "logistic"

rng = StableRNG(0)

α = 2.0
n = ceil(Int, α * d)

if model == "logistic"
    problem = ConformalAmp.Logistic(α = α, λ = λ)
else
    problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)
end

(; X, w, y) = ConformalAmp.sample_all(rng, problem, n)
@profview ConformalAmp.gamp(problem, X, y; rtol=1e-4)
@time (; xhat, vhat, ω) = ConformalAmp.gamp(problem, X, y; rtol=1e-4)

# compare the running time for amp and for erm
@profview ConformalAmp.get_cavity_means_from_gamp(problem, X, y, xhat, vhat, ω)
@time ConformalAmp.fit_leave_one_out(problem, X, y)