module ConformalAmp

using Revise
using NLSolvers
using QuadGK
using LinearAlgebra

include("problems.jl")
include("logistic_channel.jl")
include("ridge_channel.jl")
include("gamp.jl")
include("data.jl")
include("fit.jl")

end # module ConformalAmp
