

"""
    DynamicMixture

Dummy struct indicating a Dynamic Mixture Model.
"""
struct DynamicMixture end


"""
    check_dimensions(Y, F_specs, G_specs)

Internal function for checking the problem dimensions and returning them.

Returns number of clusters `k`, observational dimension `n`, state dimensions
`p = [p[1], …, p[k]]`, number of replicates `nreps`, time window length `T`,
and a vector mapping the `i`-th observation to its indexes in `Y`, in the form
of `UnitRange`'s.
"""
function check_dimensions(Y::Matrix{RT},
                          F_specs::Vector{Matrix{RT}},
                          G_specs::Vector{Matrix{RT}}) where RT <: Real

    # Number of clusters
    k = size(F_specs, 1)
    if size(G_specs, 1) != k
        throw(DimensionMismatch("F_specs' length does not match G_specs'."))
    end

    # Observational dimensions
    n = size(F_specs[1], 1)
    for j = 2:k
        if size(F_specs[j], 1) != n
            throw(DimensionMismatch("F_specs present inconsistent observational dimensions."))
        end
    end

    # Evolutional dimensions
    p = [size(G_specs[j], 1) for j = 1:k]
    for j = 1:k
        if size(G_specs[j], 2) != p[j]
            throw(DimensionMismatch("Matrices in G_specs should be square."))
        end
        if size(F_specs[j], 2) != p[j]
            throw(DimensionMismatch("F_specs present inconsistent state dimensions."))
        end
    end

    # Number of replicates
    local nreps
    try
        nreps = Int(size(Y, 2) / n)
    catch e
        if isa(e, InexactError)
            throw(DimensionMismatch("Invalid implicit number of replicates"))
        else
            throw(e)
        end
    end

    # Time window length
    T = size(Y, 1)

    # Index mapping
    index_map = [(n * (l - 1) + 1):(n * (l - 1) + n) for l = 1:nreps]

    return k, n, p, nreps, T, index_map
end


"""
    compute_weights(::DynamicMixture, Y, F_specs, G_specs, θ, ϕ, η)

Compute the membership weights of the model, essentially computing the value
from the E-step of the of the Dynamic Mixture Model.
"""
function compute_weights(::DynamicMixture,
                         Y::Matrix{RT},
                         F_specs::Vector{Matrix{RT}},
                         G_specs::Vector{Matrix{RT}},
                         θ::Vector{Matrix{RT}},
                         ϕ::Vector{Vector{RT}},
                         η::Union{Array{RT,3}, Nothing} = nothing) where RT <: Real

    k, _, _, nreps, T, index_map = check_dimensions(Y, F_specs, G_specs)

    if isnothing(η)
        η = ones(RT, T, nreps, k)
    end

    γ = similar(η)

    V = [diagm(1 ./ ϕ[j]) for j = 1:k]

    for t = 1:T
        for i = 1:nreps
            local_index = index_map[i]
            for j = 1:k
                Fⱼ = F_specs[j]
                θⱼ = θ[j]
                Vⱼ = V[j]
                γ[t,i,j] = pdf(MultivariateNormal(Fⱼ * θⱼ[t,:], Vⱼ), Y[t,local_index])
            end
            γ[t,i,:] /= sum(γ[t,i,:])
        end
    end

    return γ
end
