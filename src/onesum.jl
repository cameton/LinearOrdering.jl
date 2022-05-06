function onesum_smoothing(Ginfo, Oinfo, cols, rows)
    (; A) = Ginfo
    (; idx_to_embedding, idx_to_position, position_to_idx) = Oinfo

    if cols == (:)
        cols = 1:length(idx_to_embedding)
    end
    if rows == (:)
        rows = 1:length(idx_to_embedding)
    end

    G = SimpleGraph(Symmetric(A)) # TODO write a not insane version of this
    mask = falses(size(A, 1)...)
    mask[cols] .= true
    for row in rows
        coarse_neighbors = collect(Iterators.filter(x -> mask[x], neighbors(G, row))) # Maybe just keep this if I'm going to need it?
        sort!(coarse_neighbors; by = x -> idx_to_embedding[x])
        w = collect(A[row, coarse_neighbors])
        wsum = sum(w)
        back = collect(cumsum(w))
        idx = argmin(x -> abs(2 * back[x] - wsum), axes(back, 1))
        if idx == length(w)
            idx_to_embedding[row] = idx_to_embedding[coarse_neighbors[idx]] + 0.5
        else
            next = coarse_neighbors[idx + 1]
            cur = coarse_neighbors[idx]
            idx_to_embedding[row] = (idx_to_embedding[next] + idx_to_embedding[cur]) / 2
        end
    end

    return Oinfo
end

