"""
Test of external libraries to compute e.g. split conformal prediction
"""

using ConformalAmp
using ConformalPrediction

problem = ConformalAmp.Ridge(α = 2.0, Δ = 1.0, λ = 0.1, Δ̂ = 1.0)
d = 100

(; X, w, y) = ConformalAmp.sample_all(StableRNG(0), problem, d)