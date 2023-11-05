using SparseArrays, Graphs
using Suppressor

struct CrossingNumber <: AbstractOrderCost end

function _crosses_outgoing(W, idx2pos, w, i, j) 
    acc = zero(typeof(w))
    rows, vals = rowvals(W), nonzeros(W)
    for k in (i+1):(j-1)
        v = idx2pos.inv[k]
        for idx in nzrange(W, v)
            u, w′ = rows[idx], vals[idx]
            if idx2pos.ord[u] < i || idx2pos.ord[u] > j
                acc += w * w′
            end
        end
    end
    return acc
end

function _crosses_range(W, idx2pos, w, i, j, ran)
    acc = zero(typeof(w))
    rows, vals = rowvals(W), nonzeros(W)
    for k in ran
        v = idx2pos.inv[k]
        for idx in nzrange(W, v)
            u, w′ = rows[idx], vals[idx]
            if i < idx2pos.ord[u] < j
                acc += w * w′
            end
        end
    end
    return acc
end

_crosses_incoming(W, idx2pos, w, i, j) = _crosses_range(W, idx2pos, w, i, j, 1:(i-1)) + _crosses_range(W, idx2pos, w, i, j, (j+1):length(idx2pos.ord))

function evalorder(::CrossingNumber, W, idx2pos)
    acc = zero(eltype(W))
    rows, vals = rowvals(W), nonzeros(W)
    for v in axes(W, 2)
        for idx in nzrange(W, v)
            u, w = rows[idx], vals[idx]
            i = min(idx2pos.ord[v], idx2pos.ord[u])
            j = max(idx2pos.ord[v], idx2pos.ord[u])
            if (j - i) - 1 < length(idx2pos.ord) ÷ 2
                acc += _crosses_outgoing(W, idx2pos, w, i, j)
            else
                acc += _crosses_incoming(W, idx2pos, w, i, j)
            end
        end
    end
    return acc
end

struct RecursiveBisection <: AbstractOrderCost
    ϵ::Float64
    config::String
    workdir::String
    kahip_home::String
    sigfigs::Int
end

function ordermatrix(cost::RecursiveBisection, W, w=ones(eltype(W), size(W, 2)); config...)
    return Order(khp_tree!(cost.kahip_home, cost.workdir, cost.config, cost.ϵ, cost.sigfigs, Int[], W, w, collect(1:size(W, 2))); inv=true)
end

function partition(kahip_home, workdir, config, ϵ, sigfigs, W, w)
    n, m = length(w), nnz(W) ÷ 2
    inpath = joinpath(workdir, "tmp.graph")
    outpath = joinpath(workdir, "tmp.partition")
    open(inpath, "w") do file
        rows, vals = rowvals(W), nonzeros(W)
        write(file, "$n $m 11\n")
        for v in 1:n
            wv = trunc(Int, w[v] * 10 ^ sigfigs)
            write(file, "$wv ")
            for idx in nzrange(W, v)
                u, we = rows[idx], trunc(Int, vals[idx] * 10 ^ sigfigs)
                write(file, "$u $we ")
            end
            write(file, "\n")
        end
    end
    @suppress_out begin
        run(`$kahip_home/deploy/kaffpa $inpath --output_filename=$outpath --preconfiguration=$config --k 2 --imbalance $ϵ`)
    end
    rm(inpath)
    ret = parse.(Bool, readlines(outpath))
    rm(outpath)
    return ret
end

function khp_tree!(kahip_home, workdir, config, ϵ, sigfigs, queue, W, w, mask)
    @assert length(mask) > 0
    if length(mask) == 1
        push!(queue, mask[1])
        return queue
    end

    W′ = W[mask, mask]
    parts = partition(kahip_home, workdir, config, ϵ, sigfigs, W′, w[mask])
    p1 = mask[findall(parts)]
    p2 = mask[findall(x -> !x, parts)]
    if length(p1) == 0 || length(p2) == 0
        append!(queue, p1)
        append!(queue, p2)
        return queue
    end
    khp_tree!(kahip_home, workdir, config, ϵ, sigfigs, queue, W, w, p1)
    khp_tree!(kahip_home, workdir, config, ϵ, sigfigs, queue, W, w, p2)
    return queue
end

