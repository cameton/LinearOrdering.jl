using StatsFuns: xlogx

abstract type ConcaveStrength <: AbstractStrength end

struct OneSum <: ConcaveStrength end

(::OneSum)(x1, x2) = abs(x1 - x2)

struct ChordLength <: ConcaveStrength end

(::ChordLength)(x1, x2) = sqrt(2 - 2cospi(x1 - x2))

struct Entropy <: ConcaveStrength end

function (::Entropy)(x1, x2)
    p = abs(x1 - x2) / 2
    return -xlogx(p) - xlogx(1 - p)
end

struct Impurity <: ConcaveStrength end

function (::Impurity)(x1, x2) 
    p = abs(x1 - x2) / 2
    return p - p ^ 2
end

function _test_strength(f::ConcaveStrength, W, embedding, v, testpos)
    ret = zero(eltype(embedding))
    rows, vals = rowvals(W), nonzeros(W)
    for idx in nzrange(W, v)
        u, w = rows[idx], vals[idx]
        ret += w * f(emb[u], testpos)
    end
    return ret
end

function minimize_position(f::ConcaveStrength, W, embedding, v)
    minpos, minval = embedding[v], _test_strength(f, W, embedding, v, embedding[v])
    rows = rowvals(W)
    for idx in nzrange(W, v)
        u = rows[idx]
        newval = _test_strength(f, W, embedding, v, embedding[u])
        if newval < minval || (newval == minval && f(embedding[u], embedding[v]) < f(minpos, embedding[v]))
            minpos = embedding[u]
            minval = newval
        end
    end
    return minpos
end