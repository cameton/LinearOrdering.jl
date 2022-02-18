
sublaplacian(A, d, rows) = Matrix(Diagonal(d[rows]) - A[rows, rows])

function window_border(A, d, x, volume, window) # TODO make sure this is right
    Δx = d[window] .* x[window] .- (A[window, :] * x) 
    M = sublaplacian(A, d, window)
    b1 = volume[window]
    b2 = b1 .* x[window]
    bordered = [M b1 b2; b1' 0 0; b2' 0 0]
    b = [Δx; 0; 0]
    return \(bordered, b)[1:(end-2)]
end

function window_steps(n, windowsize, stride; rev = false) 
    if rev 
        return range(n - windowsize + 1, 1; step=-stride) 
    else
        return range(1, n - windowsize + 1; step=stride)
    end
end

function window_minimization(cost::PSum, Ginfo, Oinfo; windowsize, config, rev=false)
    (; A, d) = Ginfo
    (; order, embedding, volume) = Oinfo
    n = length(order)
    backup = similar(embedding)
    backup_order = similar(order)
    best = evalorder(cost, A, embedding)
    (; pad_percent, window_sweeps, stride_percent) = config

    accept_change = 0
    reject_change = 0
    for i in window_steps(n, windowsize, floor(Int, windowsize * stride_percent); rev=rev)
        backup .= embedding
        backup_order .= order
        invorder = invperm(order)
        window = invorder[i:(i + windowsize - 1)]
        δ = window_border(A, d, embedding, volume, window)
        embedding[window] .+= δ

        padding = ceil(Int, pad_percent * windowsize)
        padded_window = invorder[max(1, i - padding):min(n, i + windowsize - 1 + padding)]
        for j in window_sweeps
            smoothing(cost, A, d, embedding, order, volume, :, padded_window)
        end

        cur = evalorder(cost, A, embedding)
        if cur >= best
            embedding .= backup
            order .= backup_order
            reject_change += 1
        else
            best = cur
            accept_change += 1
        end
    end
    println("WindowMin Size $(size(A,1)) Winsize $windowsize Accept $accept_change Reject $reject_change Ratio $(accept_change / (accept_change + reject_change))")
    return nothing
end

function twosum_smoothing(A, d, y, cols, rows)
    y[rows] .= (A[rows, cols] * y[cols]) ./ d[rows] # TODO more efficient. I think it's correct now though
    return y
end

