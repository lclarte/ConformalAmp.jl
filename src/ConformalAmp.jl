module ConformalAmp

using LinearAlgebra
using MLJLinearModels
using NLSolvers
using QuadGK
using Revise
using Statistics

include("problems.jl")
include("logistic_channel.jl")
include("bayesian_logistic_channel.jl")
include("ridge_channel.jl")
include("pinball_channel.jl")
include("gamp.jl")

include("data.jl")
include("fit.jl")
include("conformal.jl")

end # module ConformalAmp
