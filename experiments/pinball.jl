"""
Script pour tester la pinball loss et verifier que ca fait bien de la quantile regression
"""

using ConformalAmp
using LinearAlgebra
using Plots
using ProgressBars
using StableRNGs: StableRNG
using Statistics

function plot_pinball_loss(q::Real)
    x = -5.0:0.1:5.0
    f_x = [ConformalAmp.PinballChannel.loss(0.0, x[i], q) for i in eachindex(x)]
    fprime_x = [ConformalAmp.PinballChannel.loss_der(0.0, x[i], q) for i in eachindex(x)]
    plt = plot(x, f_x, xlabel = "x", ylabel = "loss(y = 0, x)", label="loss")
    plot!(plt, x, fprime_x, label="loss derivative")
    display(plt)
end
    
function compare_erm_gamp_pinball()
    """
    Function to compare the estimator regressed using the punball loss using ERM or GAMP
    """
    rng = StableRNG(0)
    λ   = 1.0
    α   = 5.0
    d   = 1000

    problem = ConformalAmp.Pinball(λ = λ, α = α, Δ = 1.0, q = 0.9)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)

    ŵ_erm = ConformalAmp.fit(problem, X, y, ConformalAmp.ERM())
    ŵ_amp = ConformalAmp.fit(problem, X, y, ConformalAmp.GAMP(max_iter = 100, rtol = 1e-5))
    println("$(norm(ŵ_erm - ŵ_amp, 2)), $(norm(ŵ_erm, 2)), $(norm(ŵ_amp, 2))")
    # print cos angle between the two vectors
    println("$(dot(ŵ_erm, ŵ_amp) / (norm(ŵ_erm, 2) * norm(ŵ_amp, 2)))")

    plt = stephist(ŵ_erm - ŵ_amp, bins = 100, label="ŵ_erm - ŵ_amp")
    stephist!(ŵ_erm, bins = 100, label="ŵ_erm")
    display(plt)
end

function plot_prox_pinball()
    y = 1.0
    ω_range = -10.0:0.1:10.0
    x_range = copy(ω_range)
    V = 2.0
    q = 0.9

    prox_list = []
    prox_list_2 = []
    
    for ω in ω_range
        f_x = [abs2(x - ω) / 2V + ConformalAmp.PinballChannel.loss(y, x, q) for x in x_range]
        x_min = x_range[argmin(f_x)]
        push!(prox_list, x_min)

        prox = y
        if ω > y - (q - 1) * V
            prox = ω + (q - 1) * V
        elseif ω < y - q * V
            prox = ω + q * V
        end
        push!(prox_list_2, prox)
    end

    plt = plot(ω_range, prox_list, label="prox")
    plot!(ω_range, prox_list_2, label="prox 2")
    # plot!(plt, ω_range, y * ones(length(ω_range)), label="y")
    # plot!(plt, [y - q * V, y + (1 - q) * V], [y, y], seriestype = :scatter, label="q quantile")
    display(plt)   
end

compare_erm_gamp_pinball()
# plot_prox_pinball()
