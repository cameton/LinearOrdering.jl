

function _cross(idx2pos, u1, u2, v1, v2) 
    p1, p2 = idx2pos.ord[u1], idx2pos.ord[u2]
    q1, q2 = idx2pos.ord[v1], idx2pos.ord[v2] 
    return min(p1, p2) < min(q1, q2) < max(p1, p2) < max(q1, q2)
end
cross(idx2pos, u1, u2, v1, v2) = _cross(idx2pos, u1, u2, v1, v2) || _cross(idx2pos, v1, v2, u1, u2)

function crossdiff(W, idx2pos, u, v) # Obj difference moving from u to v
    rows, vals = rowvals(W), nonzeros(W)
    acc = zero(eltype(W))
    for idxu in nzrange(W, u)
        u2, wu = rows[idxu], vals[idxu]
        for idxv in nzrange(W, v)
            v2, wv = rows[idxv], vals[idxv]
            if cross(idx2pos, u, u2, v, v2)
                acc -= wu * wv
            elseif cross(idx2pos, v, u2, u, v2)
                acc += wu * wv
            end
        end
    end
    return acc
end

function wrap(i, n)
    i = mod(i, n)
    return i == 0 ? n : i
end

function ord_neighbors(idx2pos, u)
    p = idx2pos.ord[u]
    pm1 = wrap(p-1, length(idx2pos.ord))
    pp1 = wrap(p+1, length(idx2pos.ord))
    v = idx2pos.inv[pm1]
    w = idx2pos.inv[pp1]
    return (v, w)
end 

function differential(W, idx2pos, u)
    (v, w) = ord_neighbors(idx2pos, u)
    (d1, d2) = (crossdiff(W, idx2pos, u, v), crossdiff(W, idx2pos, u, w))
    return ((v, d1), (w, d2))
end

function swap!(idx2pos, u, v)
    idx2pos.ord[u], idx2pos.ord[v] = idx2pos.ord[v], idx2pos.ord[u]
    idx2pos.inv[idx2pos.ord[u]] = u
    idx2pos.inv[idx2pos.ord[v]] = v
end

function setord!(odst, osrc)
    copyto!(odst.ord, osrc.ord)
    copyto!(odst.inv, osrc.inv)
end

function greedyswaps!(idx2pos, W, v, maxswaps)
    for _ in 1:maxswaps
        ((u, d1), (w, d2)) = differential(W, idx2pos, v)
        if d1 <= d2 && d1 < 0 
            swap!(idx2pos, v, u)
        elseif d2 < d1 && d2 < 0
            swap!(idx2pos, v, w)
        else
            break
        end
    end
    return idx2pos
end

function greedyswaps!(idx2pos, W, maxswaps)
    for v in 1:size(W, 2)
        greedyswaps!(idx2pos, W, v, maxswaps)
    end
end

function minimize_swaps!(ord::OrderingProblem{CrossingNumber}, info)
    greedyswaps!(info.idx2pos, info.W, ord.config.maxswaps)
    compute_embedding!(info.embedding, info.idx2pos, info.volume)
end

function place_vertex!(placed, W, emb, v)
    ret = zero(Complex{eltype(emb)})
    rows, vals = rowvals(W), nonzeros(W)
    for idx in nzrange(W, v)
        u, w = rows[idx], vals[idx]
        if placed[u]
            ret += cispi(emb[u]) * w
        end
    end
    return wrapangle(angle(ret) / π)
end

function initializeorder!(ord::OrderingProblem{CrossingNumber}, info_fine, info_coarse)
    fill!(info_fine.embedding, zero(eltype(info_fine.embedding)))
    copyto!(@view(info_fine.embedding[info_coarse.seeds]), info_coarse.embedding)
    placed = falses(size(info_fine.W, 2))
    fill!(@view(placed[info_coarse.seeds]), true)
    for v in axes(info_fine.W, 2)
        if !placed[v]
            info_fine.embedding[v] = place_vertex!(placed, info_fine.W, info_fine.embedding, v)
            placed[v] = true
        end
    end
    sortperm!(info_fine.idx2pos, info_fine.embedding, true)
end