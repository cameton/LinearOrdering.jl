
function wrapangle(θ)
    θ %= 2
    if θ <= -1
        return θ + 2
    elseif θ > 1
        return θ - 2
    end
    return θ
end


function local_min(problem::Concave, W, emb, v)
    cmin, umin = emb[v], v
    rows, vals = rowvals(W), nonzeros(W)
    for idx in nzrange(W, v)
        u, _ = rows[idx], vals[idx]
        c = problem.cost(W, emb, v, emb[u])
        if cmin < c 
            cmin, umin = c, u
        end
    end
    return ret
end