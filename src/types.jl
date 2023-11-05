


abstract type AbstractOrderCost end

abstract type AbstractStrength end

struct OrderingProblem{C<:AbstractOrderCost, T}
    cost::C
    config::T
    rng::Xoshiro
end

struct Order
    ord::Vector{Int}
    inv::Vector{Int}
    Order(ord; inv=false) = inv ? new(invperm(ord), ord) : new(ord, invperm(ord))
    Order(ord, invord) = new(ord, invord)
end

function setord!(order, ord, inv=false)
    o1, o2 = inv ? (order.inv, order.ord) : (order.ord, order.inv)
    copyto!(o1, ord)
    invperm!(o2, o1)
    return order
end

function Base.sortperm!(order::Order, A, inv=false; args...)
    o1, o2 = inv ? (order.inv, order.ord) : (order.ord, order.inv)
    sortperm!(o1, A; args...)
    invperm!(o2, o1)
    return order
end

Base.invperm(order::Order) = Order(order.inv, order.ord)


struct Info{Tv, Ti}
    W::SparseMatrixCSC{Tv, Ti} # Symetric matrix of weights
    idx2pos::Order
    volume::Vector{Tv}
    embedding::Vector{Tv}
    seeds::Vector{Int}
end


