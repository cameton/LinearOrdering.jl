using SparseArrays
using Random: shuffle!
using Statistics: mean

function wrapangle(θ)
    θ %= 2
    return θ >= 0 ? θ : θ + 2
end

function shortsegment(θ)
    θ = wrapangle(θ)
    return θ <= 1 ? θ : θ - 2
end

struct AlgebraicDistance{T}
    refinement_factory::Function
    strength::T
    k::Int
end

struct CircularRelaxation{T<:AbstractFloat}
    ω::T
end
(f::CircularRelaxation)(x1, x2) = wrapangle(x1 + f.ω * shortsegment(x2 - x1))


struct OverRelaxation{T<:AbstractFloat}
    ω::T
end
(f::OverRelaxation)(x1, x2) = x1 + f.ω * (x2 - x1)

struct SumStrength{T<:AbstractStrength} <: AbstractOrderCost
    f::T
end

struct GaussSeidel{T}
    relaxation::T
    order::Vector{Int}
end

gs_factory(relaxation::T, rng=default_rng()) where T = x0 -> GaussSeidel{T}(relaxation, randperm(rng, length(x0)))

function refine!(f, refinement::GaussSeidel, x0)
    for i in refinement.order
        x0[i] = refinement.relaxation(x0[i], f(x0, i))
    end
    return x0
end

refine(f, refinement::GaussSeidel, x0) = refine!(f, refinement, deepcopy(x0))

struct Jacobi{T, R}
    relaxation::T
    buffer::Vector{R}
end

jac_factory(relaxation::T) where T = x0 -> Jacobi{T, eltype(x0)}(relaxation, similar(x0))

function _jacobi!(f, src, dst, rf)
    for i in eachindex(src)
        dst[i] = rf.relaxation(src[i], f(src, i))
    end
    return src
end

function refine!(f, rf::Jacobi, x0)
    buffer = similar(x0)
    for _ in 1:rf.maxiter
        if isodd(i)
            _jacobi!(f, x0, buffer, rf)
        else
            _jacobi!(f, buffer, x0, rf)
        end
    end
    if iseven(maxiter)
        copyto!(x0, buffer)
    end
    return x0
end

refine(f, rf::Jacobi, x0) = refine!(f, rf, deepcopy(x0))



using Random: default_rng

function algebraicembedding(algdist, W, rng=default_rng())
    X = 2 .* rand(rng, eltype(W), size(W, 2), algdist.k)
    refinement = algdist.refinement_factory(@view(X[:, 1]))
    for i in 1:algdist.k
        refine!(refinement, @view(X[:, i])) do x, v
            return minimize_position(algdist.strength, W, x, v)
        end
    end
    return X
end

function algebraicdistance(algdist, W, rng=default_rng())
    X = algebraicembedding(algdist, W, rng)
    k = algdist.k
    rows, vals = rowvals(W), nonzeros(W)
    Si, Sj, Sw = Int[], Int[], Vector{eltype(W)}()
    for v in axes(W, 2)
        for idx in nzrange(W, v)
            u, w = rows[idx], vals[idx]
            push!(Si, u), push!(Sj, v), push!(Sw, w / mean(algdist.strength(X[v, k], X[u, k]) for i in 1:k) )
        end
    end
    return sparse(Si, Sj, Sw, size(W)...)
end
