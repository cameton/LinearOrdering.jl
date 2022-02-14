
struct MLOrdering{C<:AbstractOrderCost, T}
    cost::C
    config::T
end

struct OrderLevel{T <: Real}
    A::SparseMatrixCSC{T}
    volume::Vector{T}
    C::Vector{Int}
    F::Vector{Int}
    order::Vector{Int}
    embedding::Vector{T}
end

enum_cumsum(iter) = enumerate(Iterators.accumulate(+, iter))

function order_embedding!(x, order, volume)
    for (i, v) in enum_cumsum(volume[idx] for idx in order)
        x[i] = v
    end
    invpermute!(x, order)
    x .-= volume ./ 2
    return x
end

using MatrixMarket

function Multilevel.coarsen!(ord::MLOrdering, level)
    (; A, volume, C, F, order, embedding) = first(level)
    strength = A # TODO compute connection strength
    Ac, P, Cc, Fc = coarsen(ord.config.coarsening, A; volume = volume, strength = strength) # TODO more generic
    Coarsening.fix_adjacency!(Ac)
    volumec = P' * volume
    orderc = sortperm(Cc; by=idx -> order[idx])
    embeddingc = similar(embedding, length(Cc))
    order_embedding!(embeddingc, orderc, volumec)
    push!(level, OrderLevel(Ac, volumec, Cc, Fc, orderc, embeddingc))
    return nothing
end


function Multilevel.doinitial(ord::MLOrdering, level)
    (; A, order) = first(level)
    return length(order) < ord.config.coarsest
end

function Multilevel.initial!(ord::MLOrdering, level)
    (; A, order, embedding, volume) = first(level)
    best = evalorder(ord.cost, A, order)
    for perm in permutations(order)
        cur = evalorder(ord.cost, A, perm)
        if cur < best
            order .= perm
            best = cur
        end
    end
    order_embedding!(embedding, order, volume)
    return nothing
end

