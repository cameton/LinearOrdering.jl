
struct SqChordLength <: AbstractStrength end

(::SqChordLength)(x1, x2) = 2 - 2cospi(x1 - x2) # Takes arguments in [0, 2]

function minimize_position(::SqChordLength, W, emb, v)
    ret = zero(Complex{eltype(emb)})
    rows, vals = rowvals(W), nonzeros(W)
    for idx in nzrange(W, v)
        u, w = rows[idx], vals[idx]
        ret += cispi(emb[u]) * w
    end
    return wrapangle(angle(ret) / π)
end
