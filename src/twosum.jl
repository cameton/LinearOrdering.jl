
function window_border(window, Δ, x, Δx, volume)
    M = Δ[window, window]
    b1 = volume[window]
    b2 = b1 .* x[window]
    bordered = [M b1 b2; b1' 0 0; b2' 0 0]
    b = [Δx[window]; 0; 0]
    return \(bordered, b)[1:(end-2)]
end

function window_steps(n, windowsize, stride; rev = false) 
    if rev 
        return range(n - windowsize + 1, 1; step=-stride) 
    else
        return range(1, n - windowsize + 1; step=stride)
    end
end

function window_minimization(cost::PSum, A, order, embedding, volume; windowsize, config, rev=false)
    n = length(order)
    backup = similar(embedding)
    backup_order = similar(order)
    Δ = laplacian_matrix(SimpleGraph(Symmetric(A))) # TODO do something better here
    best = evalorder(cost, A, order)
    (; padding, window_sweeps, stride) = config

    for i in window_steps(n, windowsize, stride; rev=rev)
        backup .= embedding
        backup_order .= order
        window = order[i:(i + windowsize - 1)]
        Δx = Δ * embedding
        δ = window_border(window, Δ, embedding, Δx, volume)
        embedding[window] .+= δ

        padded_window = order[max(1, i - padding):min(n, i + windowsize - 1 + padding)]
        for j in window_sweeps
            smoothing(cost, A, embedding, order, volume, padded_window)
        end

        cur = evalorder(cost, A, order)
        if cur > best
            embedding .= backup
            order .= backup_order
        else
            best = cur
        end
    end
    return nothing
end

function weighted_column_average(A, y, c)
    rows = rowvals(A)
    vals = nonzeros(A)
    acc_num, acc_den = 0, 0

    for idx in nzrange(A, c)
        val = vals[idx]
        acc_num += y[rows[idx]] * val
        acc_den += val
    end
    return acc_num / acc_den
end

function twosum_smoothing(A, y, cols)
    for col in cols
        y[col] = weighted_column_average(A, y, col)
    end
    return y
end

