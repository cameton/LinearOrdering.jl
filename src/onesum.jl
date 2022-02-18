

lazy_cumdiff(v, total) = Iterators.accumulate(-, v, init=total)

function onesum_smoothing(A, x, cols, rows) # TODO iterate one by one rather than all at once
    if cols == (:)
        cols = 1:length(x)
    end
    if rows == (:)
        rows = 1:length(x)
    end
    order = sortperm(cols; by = idx -> x[idx])
    total = sum(x[cols])
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
        x[row] = idx == length(cols) ? x[order[idx]] + 0.5 : (x[order[idx]] + x[order[idx+1]]) / 2
    end
    return x
end
