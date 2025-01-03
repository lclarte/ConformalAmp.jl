using ConformalAmp

using CSV
using DataFrames
using LinearAlgebra
using Plots 
using ProgressBars
using Random
using Statistics
using StableRNGs

function test_gaussian_data()
    # test on random data and random teacher
    d = 100
    n = 200
    λ = 1.0

    x = randn(n, d) / sqrt(d)
    w = randn(d)
    y = x * w

    problem = ConformalAmp.Ridge(α = n / d, λ = λ, Δ = 1.0, Δ̂ = 1.0)
    vamp = ConformalAmp.VAMP(max_iter = 100, rtol=1e-5)

    # compute the leave one out estimator for vamp
    debut = time()
    Ŵ_vamp =  ConformalAmp.fit_leave_one_out(problem, x, y, vamp)
    println("Time for VAMP : ", time() - debut)

    debut = time()
    Ŵ_erm = ConformalAmp.fit_leave_one_out(problem, x, y, ConformalAmp.ERM())
    println("Time for ERM : ", time() - debut)

    # fill a n x d matrix with copies of ŵ_vamp
    ŵ_vamp = ConformalAmp.fit(problem, x, y, vamp)
    ŵ_vamp_matrix = vcat([ŵ_vamp' for i in 1:n]...)

    print(size(ŵ_vamp_matrix))

    # plt = plot()
    # for i in 1:n
    #     scatter!(plt, Ŵ_vamp[i, :], Ŵ_erm[i, :], label = "", marker= :circle)
    #     scatter!(plt, ŵ_vamp_matrix[i, :], Ŵ_erm[i, :], label = "", marker = :cross)
    # end
    # plot!(plt, [minimum(Ŵ_vamp), maximum(Ŵ_vamp)], [minimum(Ŵ_vamp), maximum(Ŵ_vamp)], label = "y = x", color = :black)
    # display(plt)

     # plot the histogram of the norm of the difference between the two estimators
    norm_diff = [norm(Ŵ_vamp[i, :] - Ŵ_erm[i, :]) for i in 1:n]
    ref_norm_diff = [norm(ŵ_vamp_matrix[i, :] - Ŵ_erm[i, :]) for i in 1:n]
    stephist(norm_diff, title = "Histogram of the norm of the difference between the two estimators", xlabel = "Norm of the difference", ylabel = "Frequency", label="VAMP l.o.o - ERM l.o.o", bins = 100)
    stephist!(ref_norm_diff, title = "Histogram of the norm of the difference between the two estimators", xlabel = "Norm of the difference", ylabel = "Frequency", label="VAMP - ERM l.o.o", bins = 100)

    # xtest = rand(d) / sqrt(d)
    # @time ConformalAmp.get_confidence_interval(problem, x, y, xtest, ConformalAmp.FullConformal(δy_range = 0.0:0.05:2.5, coverage = 0.9), vamp)
    # @time ConformalAmp.get_confidence_interval(problem, x, y, xtest, ConformalAmp.FullConformal(δy_range = 0.0:0.05:2.5, coverage = 0.9), ConformalAmp.ERM())
end

function test_gaussian_data_2()
    d = 500
    n = 1000
    λ = 1.0

    x = randn(n, d) / sqrt(d)
    w = randn(d)
    y = x * w

    problem = ConformalAmp.Ridge(α = n / d, λ = λ, Δ = 1.0, Δ̂ = 1.0)
    vamp = ConformalAmp.VAMP(max_iter = 100, rtol=1e-5)

    x̂_1, α_1, γ_1, x̂_2, α_2, γ_2 = ConformalAmp.vamp(problem, x, y; n_iterations = 100, rtol = 1e-5)
    Ŵ_vamp = ConformalAmp.iterate_loo_vamp_ridge(x, y, problem.Δ̂, x̂_1, α_1, γ_1, x̂_2, α_2, γ_2)'
    Ŵ_erm = ConformalAmp.fit_leave_one_out(problem, x, y, ConformalAmp.ERM())

    ŵ_vamp = ConformalAmp.fit(problem, x, y, vamp)
    ŵ_vamp_matrix = vcat([ŵ_vamp' for i in 1:n]...)

    norm_diff = [norm(Ŵ_vamp[i, :] - Ŵ_erm[i, :]) for i in 1:n]
    ref_norm_diff = [norm(ŵ_vamp_matrix[i, :] - Ŵ_erm[i, :]) for i in 1:n]

    stephist(norm_diff, title = "Histogram of the norm of the difference between the two estimators", xlabel = "Norm of the difference", ylabel = "Frequency", label="VAMP l.o.o - ERM l.o.o", bins = 100)
    stephist!(ref_norm_diff, title = "Histogram of the norm of the difference between the two estimators", xlabel = "Norm of the difference", ylabel = "Frequency", label="VAMP - ERM l.o.o", bins = 100)

end

function test_superconductivity()
    data = CSV.read("experiments/misc/superconductivity.csv", DataFrame)

    n_max = 600
    n = min(size(data, 1), n_max) # use a reasonable amount of data
    # shuffle the data randomly 
    data = data[shuffle(1:size(data, 1)), :]
    y = data[1:n, end]
    X = data[1:n, 1:end-1]

    # convert to matrix and normalize the data

    X = Matrix(X)
    y = Vector(y)
    X = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
    X ./= sqrt(size(X, 2))
    y = (y .- mean(y)) / std(y)

    d = size(X, 2)

    problem = ConformalAmp.Ridge(α = n / d, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

    # do a train test split
    n_train = Integer(n * 0.5)
    n_test  = n - n_train

    X_train, X_test = X[1:n_train, :], X[n_train+1:end, :]
    y_train, y_test = y[1:n_train], y[n_train+1:end]

    fcp = ConformalAmp.FullConformal(δy_range = 0.0:0.005:6.0, coverage = 0.95)
    vamp = ConformalAmp.VAMP(max_iter = 100, rtol = 1e-3)

    ci_list_vamp = []
    for xtest in ProgressBar(eachrow(X_test))
        # compute the confidence intervals for each of them  
        ci_vamp = ConformalAmp.get_confidence_interval(problem, X_train, y_train, xtest, fcp, vamp)
        # push the min and max of the list
        push!(ci_list_vamp, (minimum(ci_vamp), maximum(ci_vamp)))
    end

    # compute the coverage of the VAMP
    coverage_vamp = mean([ci_list_vamp[i][1] <= y_test[i] <= ci_list_vamp[i][2] for i in 1:n_test])
    println("Coverage of VAMP : ", coverage_vamp)
end

function test_boston()
    data = CSV.read("experiments/misc/boston.csv", DataFrame; missingstring=["NA"])
    x = data[:, 1:end-1]
    y = data[:, end]

    # normalize the data
    # replace missing values by the mean
    x = Matrix(x)
    y = Vector(y)

    
    n, d = size(x)
    ntrain = Integer(floor(0.8 * n))
    ntest = n - ntrain
    
    x = coalesce.(x, 0)
    x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
    x ./= sqrt(d)
    y = (y .- mean(y)) / std(y)

    xtrain, xtest = x[1:ntrain, :], x[ntrain+1:end, :]
    ytrain, ytest = y[1:ntrain], y[ntrain+1:end]

    problem = ConformalAmp.Ridge(α = n / d, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)

    ci_vamp_list = []
    for x in ProgressBar(eachrow(xtest))
        ci = ConformalAmp.get_confidence_interval(problem, xtrain, ytrain, x, ConformalAmp.FullConformal(δy_range = 0.0:0.005:2.0, coverage = 0.85), ConformalAmp.VAMP(max_iter = 100, rtol = 1e-3))
        push!(ci_vamp_list, (minimum(ci), maximum(ci)))
    end

    # compute the lgnth of the confidence intervals
    lengths = [ci_vamp_list[i][2] - ci_vamp_list[i][1] for i in 1:ntest]
    plt = stephist(lengths, title = "Histogram of the confidence intervals", xlabel = "Confidence interval", ylabel = "Frequency")
    display(plt)

    # compute the coverage
    coverage_vamp = mean([ci_vamp_list[i][1] <= ytest[i] <= ci_vamp_list[i][2] for i in 1:ntest])
    println("Coverage of VAMP : ", coverage_vamp)
end

function jaccard_index(a1::Real, b1::Real, a2::Real, b2::Real)
    intersection = max(0, min(b1, b2) - max(a1, a2))
    union = max(b1, b2) - min(a1, a2)
    return intersection / union
end

function compare_vamp_gamp()
    rng = StableRNG(0)

    range = [ 500 ]
    for d in (range)
        vamp = ConformalAmp.VAMP(max_iter = 100, rtol = 1e-4)
        gamp = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
        
        problem = ConformalAmp.Ridge(α = 0.5, λ = 1.0, Δ = 1.0, Δ̂ = 1.0)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        
        ntest = 20
        xtest_array = ConformalAmp.sample_data_any_n(rng, d, ntest)

        ci_vamp_list = []
        ci_gamp_list = []

        for x in (eachrow(xtest_array))
            ci_vamp = ConformalAmp.get_confidence_interval(problem, X, y, x, ConformalAmp.FullConformal(δy_range = 0.0:0.02:3.0, coverage = 0.85), vamp)
            ci_gamp = ConformalAmp.get_confidence_interval(problem, X, y, x, ConformalAmp.FullConformal(δy_range = 0.0:0.02:3.0, coverage = 0.85), gamp)
            push!(ci_vamp_list, (minimum(ci_vamp), maximum(ci_vamp)))
            push!(ci_gamp_list, (minimum(ci_gamp), maximum(ci_gamp)))
        end

        # compute the jaccard indices between the two sets of confidence intervals
        jaccard_indices = [jaccard_index(ci_vamp_list[i][1], ci_vamp_list[i][2], ci_gamp_list[i][1], ci_gamp_list[i][2]) for i in 1:ntest]
        println("Jaccard index at dimension $d: ", mean(jaccard_indices))
    end

end

test_gaussian_data_2()
# test_boston()
# test_superconductivity()
# compare_vamp_gamp()