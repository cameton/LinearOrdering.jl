

function onesum_smoothing(A, idx_to_embedding, cols, rows) # TODO iterate one by one rather than all at once
    if cols == (:)
        cols = 1:length(idx_to_embedding)
    end
    if rows == (:)
        rows = 1:length(idx_to_embedding)
    end
    order = invperm(sortperm(cols; by = idx -> idx_to_embedding[idx])) # TODO should this be like this?
    total = sum(idx_to_embedding[cols])
    best = total
    for row in rows # TODO make more efficient, remove nested loop
        itersum = lazy_cumsum(A[row, col] for col in order)
        iterdiff = lazy_cumdiff((A[row, col] for col in order), total)
        idx = length(cols)
        for (i, (vsum, vdiff)) in enumerate(Iterators.zip(itersum, iterdiff))
            if abs(vsum - vdiff) < best
                best = abs(vsum - vdiff)
                idx = i
            end
        end
        idx_to_embedding[row] = idx == length(cols) ? idx_to_embedding[order[idx]] + 0.5 : (idx_to_embedding[order[idx]] + idx_to_embedding[order[idx+1]]) / 2
    end
    println("PRE INFLATION $(evalorder(PSum(1), A, idx_to_embedding))")
    return idx_to_embedding
end
