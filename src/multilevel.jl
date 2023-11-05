using LinearAlgebra

function ordergraph(cost, G, volume=normalize!(ones(Float64, nv(G)), 1); config...)
    @assert is_connected(G)
    return ordermatrix(cost, 1.0 * adjacency_matrix(G), volume; config...)
end

# windowsizes, compat_sweeps, window_sweeps, stride, gauss_sweeps, coarsening, pad_percent
function ordermatrix(cost, W, volume=normalize!(ones(eltype(W), size(W, 2)), 1); config...)
    config = NamedTuple(config)
    rng = Xoshiro(config.seed)
    ord = OrderingProblem(cost, config, rng)
    idx2pos = Order(randperm(rng, size(W, 1)))
    embedding = compute_embedding(idx2pos, volume)
    info = Info(W, idx2pos, volume, embedding, Int[])
    levels = Stack{typeof(info)}()
    push!(levels, info)
    Multilevel.cycle!(ord, levels)
    return idx2pos
end

compute_embedding(idx2pos, volume) = compute_embedding!(similar(volume), idx2pos, volume)
function compute_embedding!(embedding, idx2pos, volume)
    acc = zero(eltype(volume))    
    for idx in idx2pos.inv 
        acc += volume[idx]
        embedding[idx] = acc
        acc += volume[idx]
    end
    return embedding
end

function coarse_info(W, P, seeds, volume, idx2pos)
    Wc = P' * (W * P)
    vc = P' * volume
    invoc = zeros(eltype(idx2pos.ord), length(vc))
    copyto!(invoc, idx2pos.ord[seeds])
    coarse_idx2pos = Order(sortperm(invoc); inv=true)
    ec = compute_embedding(coarse_idx2pos, vc)
    return Info(Wc, coarse_idx2pos, vc, ec, seeds)
end

function Multilevel.descend!(ord::OrderingProblem, levels)
    info = first(levels)
    strength = info.W # algebraicdistance(ord.config.algdist, info.W, ord.rng)
    seeds = Coarsening.coarseseeds(ord.config.coarsening, strength; volume=info.volume)
    P = coarseprojection(info.W, seeds, ord.config.order)

    push!(levels, coarse_info(info.W, P, seeds, info.volume, info.idx2pos))
    return nothing
end

function Multilevel.is_lowest(ord::OrderingProblem, levels)
    info = first(levels)
    return size(info.W, 1) < ord.config.coarsest
end

using Combinatorics: permutations

function Multilevel.lowest!(ord::OrderingProblem, levels)
    info = first(levels)

    mincost = evalorder(ord.cost, info.W, info.idx2pos)
    minorder = info.idx2pos
    for perm in permutations(1:size(info.W, 2), size(info.W, 2))
        testorder = Order(perm)
        cost = evalorder(ord.cost, info.W, testorder)
        if cost < mincost
            minorder = testorder
        end
    end
    info.idx2pos.ord[:] .= minorder.ord[:]
    info.idx2pos.inv[:] .= minorder.inv[:]
    # @debug debug_level_info(level_id, Ginfo, Oinfo)

    # @debug """
    # ########## Initial Level $level_id 
    # Pre: $preeval
    # Post: $posteval
    # """

    return nothing
end

function compatible!(f, refinement, embedding, fixed)
    refine!(refinement, embedding) do x, v
        return insorted(fixed, v) ? x[v] : f(x, v)
    end
end

function Multilevel.ascend!(ord::OrderingProblem, levels)
    info_coarse = pop!(levels)
    info_fine = first(levels)

    initializeorder!(ord, info_fine, info_coarse)
    for i in 1:10
        minimize_swaps!(ord, info_fine)
    end
 
end
