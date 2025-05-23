module ConformalAmp

using LinearAlgebra
using MLJLinearModels
using NLSolvers
using QuadGK
using Revise
using Statistics
using RCall

include("problems.jl")
include("logistic_channel.jl")
include("bayesian_logistic_channel.jl")
include("ridge_channel.jl")
include("pinball_channel.jl")

include("l1_prior.jl")
include("l2_prior.jl")

include("gamp.jl")

include("data.jl")
include("fit.jl")
include("vamp.jl")
include("conformal.jl")
include("splitconformal.jl")
include("conformalquantile.jl")

end # module ConformalAmp
