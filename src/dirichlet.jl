
"""
    forward_filter(Z, δ, c₀)

Performs the forward filtering algorithm for a Dirichlet Evolutional Process, as
described in Fonseca & Ferreira (2017), and returns the online Dirichlet
parameters `c[1], …, c[T]`.
"""
function forward_filter(Z::Matrix{<: Integer},
                        δ::RT,
                        c₀::Vector{RT} = fill(RT(0.1), size(Z,2))) where RT <: Real

    T, k = size(Z)
    c = Matrix{RT}(undef, T, k)

    c[1,:] = δ * c₀ + Z[1,:]
    for t = 2:T
        c[t,:] = δ * c[t-1,:] + Z[t,:]
    end

    return c
end


"""
    backwards_sampler(c, δ)

Performs the backwards sampling algorithm for a Dirichlet Evolutional Process,
as described in Fonseca & Ferreira (2017), and returns samples `η[1], …, η[T]`
from the posterior distribution.
"""
function backwards_sampler(c::Matrix{RT}, δ::RT) where RT <: Real

    T = size(c, 1)
    η = similar(c)

    η[T,:] = rand(Dirichlet(c[T,:]))
    for t = T-1:-1:1
        cₛ = sum(c[t,:])
        S = rand(Beta(δ * cₛ, (1 - δ) * cₛ))
        u = rand(Dirichlet((1 - δ) * c[t,:]))
        η[t,:] = S * η[t+1,:] + (1 - S) * u
    end

    return η
end


"""
    moddirichlet_mean(c, a, b)

Internal function computing the mean for a Mod-Dirichlet.
"""
@inline function moddirichlet_mean(c::Vector{RT},
                                   a::Vector{RT},
                                   b::Vector{RT}) where RT <: Real

    return (b - a) .* (c / sum(c)) + a
end


"""
    moddirichlet_params(c, δ, η)

Internal function to return the parameters for the Mod-Dirichlet distribution.
"""
function moddirichlet_params(c::Vector{RT},
                             δ::RT,
                             η::Vector{RT}) where RT <: Real

    k = size(η, 1)

    # Compute mode for S

    cₛ = sum(c)
    α = δ * cₛ
    β = (1 - δ) * cₛ

    if (α < 1.) & (β < 1.)
        throw(ArgumentError("Invalid mode for S"))
    elseif α <= 1.
        S = 0
    elseif β <= 1.
        S = 1
    else
        S = (α - 1) / (α + β - 2)
    end

    # Compute parameters as function of the mode of S

    c = (1 - δ) * c
    a = S * η
    b = (1 - S) * ones(k) + S * η

    return c, a, b
end


"""
    backwards_estimator(c, δ)

Similar to `backwards_sampler`, but obtains modes in a backwards step instead
of samples.
"""
function backwards_estimator(c::Matrix{RT}, δ::RT) where RT <: Real

    T, k = size(c)
    η = similar(c)

    # For the last eta's distribution we can do it directly since it's a known
    # Dirichlet(c[T]).

    η[T,:] = moddirichlet_mean(c[T,:], zeros(k), ones(k))

    # For the other time instants we need to find the Mod-Dirichlet parameters
    # conditional to the previous mode being a known value.

    for t = T-1:-1:1
        cₜ, aₜ, bₜ = moddirichlet_params(c[t,:], δ, η[t+1,:])
        η[t,:] = moddirichlet_mean(cₜ, aₜ, bₜ)
    end

    return η
end
