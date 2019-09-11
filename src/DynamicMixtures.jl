module DynamicMixtures

using Distributions: Beta, Dirichlet, Multinomial, MultivariateNormal, pdf
using DynamicLinearModels
using LinearAlgebra
using StatsBase

include("dirichlet.jl")
include("dynamic.jl")

end # module
