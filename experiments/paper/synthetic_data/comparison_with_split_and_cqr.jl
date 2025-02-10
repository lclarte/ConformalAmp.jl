"""
Compare with split conformal prediction and conformalized quantile regression
"""

using ConformalAmp
using ProgressBars
using StableRNGs


using Statistics
# generate the data 

α = 2.0
λ = 1.0
d = 50

problem = ConformalAmp.Ridge(α = α, Δ = 1.0, λ = λ, Δ̂ = 1.0)

ntest = 1
coverage = 0.9

fcp = ConformalAmp.FullConformal(coverage = coverage, δy_range = 0.0:0.05:5.0)

# ========================= BELOW IS EXACT FCP ======================= 
rng = StableRNG(10)
ntest_fcp = 10
if true
    coverages_fcp = []
    interval_sizes_fcp = []
    
    for i in ProgressBar(1:ntest_fcp)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
        ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)
        intervals_fcp = []
        sizes_fcp     = []
        
        for x in eachrow(Xtest)
            interval_fcp = ConformalAmp.get_confidence_interval(problem, X, y, x, fcp, ConformalAmp.ERM())
            push!(intervals_fcp, interval_fcp)
            push!(sizes_fcp, maximum(interval_fcp) - minimum(interval_fcp))
        end
        
        coverage_fcp = mean((ytest .>= minimum.(intervals_fcp)) .& (ytest .<= maximum.(intervals_fcp)))
        push!(coverages_fcp, coverage_fcp)
        push!(interval_sizes_fcp, mean(sizes_fcp))
    end
end

# ========================= BELOW IS TAYLOR-AMP ======================= 
rng = StableRNG(10)
if true # put to false if we don't want to test it 
    coverages_amp = []
    interval_sizes_amp = []
    
    for i in ProgressBar(1:10^3)
        (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
        Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
        ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)
        intervals_amp = []
        sizes_amp     = []
        
        for x in eachrow(Xtest)
            interval_amp = ConformalAmp.get_confidence_interval(problem, X, y, x, fcp, ConformalAmp.GAMP(rtol = 1e-3, max_iter = 100))
            push!(intervals_amp, interval_amp)
            push!(sizes_amp, maximum(interval_amp) - minimum(interval_amp))
        end
        
        coverage_amp = mean((ytest .>= minimum.(intervals_amp)) .& (ytest .<= maximum.(intervals_amp)))
        push!(coverages_amp, coverage_amp)
        push!(interval_sizes_amp, mean(sizes_amp))
    end
end

# ========================= BELOW IS CONFORMALIZED QUANTILE REGRESSION =======================
rng = StableRNG(10)
coverages_cqr = []
interval_sizes_cqr = []
for i in ProgressBar(1:10^4)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)

    lowers, uppers = ConformalAmp.get_intervals_qcp(X, y, Xtest, fcp.coverage, problem.λ, rng)

    # compute the coverage
    coverage_cqr = mean((ytest .>= lowers) .& (ytest .<= uppers))
    push!(coverages_cqr, coverage_cqr)
    push!(interval_sizes_cqr, mean(uppers - lowers))
end

# ========================= BELOW IS SPLIT CONFORMAL REGRESSION =======================
rng = StableRNG(10)
coverages_scp = []
interval_sizes_scp = []
for i in ProgressBar(1:10^4)
    (; X, w, y) = ConformalAmp.sample_all(rng, problem, d)
    Xtest = ConformalAmp.sample_data_any_n(rng, d, ntest)
    ytest = ConformalAmp.sample_labels(rng, problem, Xtest, w)

    lowers, uppers = ConformalAmp.get_confidence_interval(problem, X, y, Xtest, ConformalAmp.SplitConformal(coverage = coverage), ConformalAmp.ERM())

    # compute the coverage
    coverage_scp = mean((ytest .>= lowers) .& (ytest .<= uppers))
    push!(coverages_scp, coverage_scp)
    push!(interval_sizes_scp, mean(uppers - lowers))
end

println("α = $α, λ = $λ, d = $d")
println("CQP : ")
println("Coverage : $(mean(coverages_cqr)) pm $(std(coverages_cqr))")
println("Interval size : $(mean(interval_sizes_cqr)) pm $(std(interval_sizes_cqr))")
println("========")
println("SCP : ")
println("Coverage : $(mean(coverages_scp)) pm $(std(coverages_scp))")
println("Interval size : $(mean(interval_sizes_scp)) pm $(std(interval_sizes_scp))")
println("========")
println("AMP : ")
println("Coverage : $(mean(coverages_amp)) pm $(std(coverages_amp))")
println("Interval size : $(mean(interval_sizes_amp)) pm $(std(interval_sizes_amp))")