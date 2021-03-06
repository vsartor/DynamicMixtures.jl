

"""
    DynamicMixture

Dummy struct indicating a Dynamic Mixture Model for Time-Series data.
"""
struct DynamicMixture end


"""
    StaticMixture

Dummy struct indicating a Static Mixture Model for Time-Series data.
"""
struct StaticMixture end


"""
    check_dimensions(Y, F_specs, G_specs)

Internal function for checking the problem dimensions and returning them.

Returns observational dimension `n`, number of replicates `nreps`, state
dimensions `p = [p[1], …, p[k]]`, time-window size `T`, number of clusters `k`,
and a vector mapping the `i`-th observation to its indexes in `Y`, in the form
of `UnitRange`'s. Returning a tuple `n, nreps, p, T, k, index_map`.
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

    return n, nreps, p, T, k, index_map
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

    _, nreps, _, T, k, index_map = check_dimensions(Y, F_specs, G_specs)

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


"""
    initialize(::DynamicMixture, Y, F_specs, G_specs)

Initializes the model parameters `θ`, `ϕ`, and `η`. Note that to avoid problems
regarding bad relative scales of observational variances, the `ϕ` are
initialized to `ones`.
"""
function initialize(model::Union{DynamicMixture, StaticMixture},
                    Y::Matrix{RT},
                    F_specs::Vector{Matrix{RT}},
                    G_specs::Vector{Matrix{RT}}) where RT <: Real

    n, nreps, p, T, k, index_map = check_dimensions(Y, F_specs, G_specs)

    ϕ = [ones(n) for _ = 1:k]
    θ = Vector{Matrix{RT}}(undef, k)

    # Step 1: Initialize algorithm-specific variables
    centroids = Vector{Int}(undef, 0)
    candidates = collect(1:nreps)
    distances = Matrix{RT}(undef, nreps, 0)

    # Step 2: Pick the first centroid uniformly at random
    push!(centroids, sample(candidates))

    # Step 3: Pick further centroids based on distance
    for _ = 2:k
        # Remove last picked centroid from candidates
        deleteat!(candidates, findfirst(x -> x == centroids[end], candidates))

        # Add to `distances` the distance between every observation and the last
        # picked centroid
        recent_centroid = Y[:,index_map[centroids[end]]]
        local_distances = [sum(abs.(Y[:,index_map[i]] - recent_centroid))  for i = 1:nreps]
        distances = hcat(distances, local_distances)

        # Get each candidate's biggest distance to the centroids
        weights = Weights(maximum(distances, dims=2)[candidates,1])

        # Obtain the new centroid weighted by the its maximum distances
        push!(centroids, sample(candidates, weights))
    end

    # Step 4: Initialize each cluster's parameters based on MAP estimation

    # TODO: Current ordering is assumed to be arbitrary, which is only true if
    # all F_j and G_j are the same. When it isn't adjust all k models for all k
    # centroids and pick the highest likelihood candidate for each model.

    for j = 1:k
        θ[j] = estimate(Y[:,index_map[centroids[j]]], F_specs[j], G_specs[j], 0.7)[1]
    end

    # Step 5: Compute weights

    η = compute_weights(model, Y, F_specs, G_specs, θ, ϕ)

    return θ, ϕ, η
end


"""
    dynamic_mixture(Y, F_specs, G_specs, δ[; numit, mnumit, ϵ, M])

Uses Stochastic Expectation Maximization to obtain MAP estimates for the Dynamic
Mixture of Dynamic Linear Models.

* `Y` is the matrix of observations with `T` rows and `n * nreps` columns;
* `F_specs` is a vector of observational matrices;
* `G_specs` is a vector of evolutional matrices;
* `δ` is the vector of discount factors for each observational unit;
* `numit` is the number of iterations;
* `M` is the number of simulations in the stochastic step of the Stochastic Expectation Maximization.

Returns the estimated `η`, `θ`, `ϕ` and the implied `W`.
"""
function dynamic_mixture(Y::Matrix{RT},
                         F_specs::Vector{Matrix{RT}},
                         G_specs::Vector{Matrix{RT}},
                         δ::Vector{RT};
                         numit::Int = 10,
                         M::Int = 200) where RT <: Real

    _, nobs, _, T, k, _ = check_dimensions(Y, F_specs, G_specs)

    θ, ϕ, η = initialize(DynamicMixture(), Y, F_specs, G_specs)

    c₀ = ones(k)

    mc_observations = Matrix{Int}(undef, T, k)
    mc_estimates = Matrix{RT}(undef, T, k)

    for _ = 1:numit
        # Step 1: Expectation-step

        weights = compute_weights(DynamicMixture(), Y, F_specs, G_specs, θ, ϕ, η)

        # Step 2: Maximize weights for each observation

        for i = 1:nobs
            # Simulate M observations and obtain the mean of all M estimates
            fill!(mc_estimates, 0.)
            for _ = 1:M
                for t = 1:T
                    mc_observations[t,:] = rand(Multinomial(1, η[t,i,:]))
                end
                c = forward_filter(mc_observations, δ[i], c₀)
                mc_estimates += backwards_estimator(c, δ[i]) / M
            end
            η[:,i,:] = mc_estimates
        end

        # Step 3: Maximize the cluster parameters

        for j = 1:k
            θ[j], V = estimate(Y, F_specs[j], G_specs[j], η[:,:,j], 0.7)
            ϕ[j] = 1 ./ diag(V)
        end

    end

    W = [evolutional_covariances(Y, F_specs[j], G_specs[j], Symmetric(diagm(1 ./ ϕ[j])), η[:,:,j], 0.7) for j = 1:k]

    return η, θ, ϕ, W
end
