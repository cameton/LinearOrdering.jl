
function shiftto!(x, src, dst, segsize)
    buf = similar(x, segsize) # TODO persistent buffer allocation
    copyto!(buf, 1, x, src, segsize)
    if src < dst
        # println("SIZE $(size(x)) SRC $(src + segsize) DST $src SEG $(dst - src + 1)")
        copyto!(x, src, x, src + segsize, dst - src)
    else
        copyto!(x, dst + segsize, x, dst, src - dst)
    end
    copyto!(x, dst, buf, 1, segsize)
    return x
end

function positionshift!(Oinfo, src, dst, segsize)
    (; position_to_idx, idx_to_position, idx_to_embedding, volume) = Oinfo

    shiftto!(position_to_idx, src, dst, segsize)

    lo = min(src, dst)
    up = lo + abs(src - dst) + segsize - 1
    # println("SRC $src DST $dst LO $lo UP $up")
    init = zero(eltype(idx_to_embedding))
    if lo > 1
        lom1idx = position_to_idx[lo - 1]
        init = idx_to_embedding[lom1idx] + volume[lom1idx] / 2
    end
    newembedding = lazy_cumsum((volume[position_to_idx[pos]] for pos in lo:up); init = init)

#   println("LO $lo UP $up")
    for (pos, v) in Iterators.zip(lo:up, newembedding)
        idx = position_to_idx[pos]
#       println("POS $pos IDX $idx")
        idx_to_embedding[idx] = v - volume[idx] / 2
        idx_to_position[idx] = pos
    end
    return Oinfo
end

# TODO add max iter
function node_by_node!(cost, Ginfo, Oinfo, k) # TODO FIX THIS
    (; A) = Ginfo
    (; idx_to_embedding, idx_to_position, position_to_idx, volume) = Oinfo

    n = length(idx_to_embedding)
    buf = zeros(1)
    best = evalorder(cost, A, idx_to_embedding)
    backup = deepcopy(Oinfo)
#   println(position_to_idx)
#   println(idx_to_position)
#   println(idx_to_embedding)
#   println()
 
    acc = 0
    accepted = 0
    movable_idx = trues(size(A, 2))
    while any(movable_idx)
        for col in findall(movable_idx)
            movable_idx[col] = false
            src = idx_to_position[col]
            lo = max(1, src - k)
            up = min(n, src + k)
            for dst in lo:up
    #           println("Col $col Src $src Dst $dst Eval $(evalorder(cost, A, idx_to_embedding))")
    #           println(position_to_idx)
    #           println(idx_to_position)
    #           println(idx_to_embedding)
    #           println()
                positionshift!(Oinfo, src, dst, 1)
                test = evalorder(cost, A, idx_to_embedding)
                if test <= best
                    if test < best 
                        movable_idx[col] = true
                    end
                    best = test
                    copyorder!(backup, Oinfo)
                    accepted += 1
                end
                src = dst
                acc += 1
            end
        end
    end
#   println("ACC $acc ACCEPTED $accepted RATIO $(accepted / acc)")
    copyorder!(Oinfo, backup)
    return Oinfo
end

