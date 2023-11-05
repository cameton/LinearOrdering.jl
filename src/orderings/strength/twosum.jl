struct TwoSum <: AbstractStrength end

(::TwoSum)(x1, x2) = abs(x1 - x2) ^ 2 # Takes arguments in [0, 2)

function minimize_position(::TwoSum, W, emb, v)
    num, den = zero(eltype(emb)), zero(eltype(emb))
    rows, vals = rowvals(W), nonzeros(W)

    for idx in nzrange(W, v)
        u, w = rows[idx], vals[idx]
        num += w * emb[u]
        den += w
    end
    return num / den
end