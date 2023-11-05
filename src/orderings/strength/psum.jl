struct PSum{T<:Real} <: AbstractStrength
    p::T
end

(f::PSum)(x1, x2) = abs(x1 - x2) ^ f.p # Takes arguments in [0, 2]

function minimize_position(::PSum, W, emb, v)
    return nothing # TODO
end
