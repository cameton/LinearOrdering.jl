
struct MLOrdering{C<:AbstractOrderCost, T}
    cost::C
    config::T
end

struct OrderInfo{T <: Real}
    idx_to_embedding::Vector{T}
    position_to_idx::Vector{Int}
    idx_to_position::Vector{Int}
    volume::Vector{T}
end

mutable struct GraphInfo{T <: Real}
    A::SparseMatrixCSC{T}
    d::Vector{T}
    C::Union{Nothing, Vector{Int}}
    F::Union{Nothing, Vector{Int}}
end

struct OrderLevel{T <: Real}
    Ginfo::GraphInfo{T}
    Oinfo::OrderInfo{T}
end

function inflate_embedding!(idx_to_embedding, position_to_idx, volume)
    for (i, v) in enumerate(lazy_cumsum(volume[idx] for idx in position_to_idx))
        idx_to_embedding[i] = v
    end
    invpermute!(idx_to_embedding, position_to_idx)
    idx_to_embedding .-= volume ./ 2
    return idx_to_embedding
end
function inflate_embedding!(Oinfo)
    (; position_to_idx, idx_to_position, idx_to_embedding, volume) = Oinfo
    sortperm!(position_to_idx, idx_to_embedding)
    invperm!(idx_to_position, position_to_idx)
    inflate_embedding!(idx_to_embedding, position_to_idx, volume)
    return Oinfo
end

function init_coarse_order(C, P, Oinfo::OrderInfo{T}) where {T}
    (; idx_to_position, volume) = Oinfo

    position_to_coarse_idx = sortperm(C; by = idx -> idx_to_position[idx])
    coarse_idx_to_position = invperm(position_to_coarse_idx)
    coarse_idx_to_embedding = collect(T, coarse_idx_to_position)
    volumec = P' * volume

    cOinfo = OrderInfo(coarse_idx_to_embedding, position_to_coarse_idx, coarse_idx_to_position, volumec)

    return inflate_embedding!(cOinfo)
end

function Multilevel.coarsen!(ord::MLOrdering, level)
    (; Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; position_to_idx, idx_to_embedding, volume) = Oinfo
    dropzeros!(A) # TODO necessary?

    testG = SimpleGraph(Symmetric(A))
    print_graph_info("Fine", testG)
    print_volume_info("Fine", volume)
    print_adj_info("Fine", A)

    strength = A # TODO compute connection strength
    Ac, P, Ginfo.C, Ginfo.F = coarsen(ord.config.coarsening, A; volume = volume, strength = strength) # TODO more generic
    Coarsening.fix_adjacency!(Ac)
    dropzeros!(Ac)

    cGinfo = GraphInfo(Ac, sumcols(Ac), nothing, nothing)
    cOinfo = init_coarse_order(Ginfo.C, P, Oinfo)

    testG = SimpleGraph(Symmetric(Ac))

    println("")
    println("-----------------------------------")
    println("")

    testGc = SimpleGraph(Ac)
    print_graph_info("Coarse", testGc)
    print_volume_info("Coarse", cOinfo.volume)
    print_adj_info("Coarse", Ac)


    push!(level, OrderLevel(cGinfo, cOinfo))
    return nothing
end


function Multilevel.doinitial(ord::MLOrdering, level)
    (; Oinfo) = first(level)
    return length(Oinfo.idx_to_embedding) < ord.config.coarsest
end

function copyorder!(dstorder, srcorder)
    copyto!(dstorder.position_to_idx, srcorder.position_to_idx)
    copyto!(dstorder.idx_to_position, srcorder.idx_to_position)
    copyto!(dstorder.idx_to_embedding, srcorder.idx_to_embedding)
    copyto!(dstorder.volume, srcorder.volume)
    return dstorder
end

function Multilevel.initial!(ord::MLOrdering, level)
    (; Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; idx_to_embedding) = Oinfo

    best = evalorder(ord.cost, A, idx_to_embedding)
    backup = deepcopy(Oinfo)
    println("Coarsest PreEval $best")
    for perm in permutations(1:length(idx_to_embedding))
        idx_to_embedding .= perm
        inflate_embedding!(Oinfo) # TODO see if it's necessary to do the full process
        cur = evalorder(ord.cost, A, idx_to_embedding)
        if cur < best
            copyorder!(backup, Oinfo)
            best = cur
        end
    end
    copyorder!(Oinfo, backup)
    println("Coarsest PostEval $best")
    return nothing
end

function shift!(buf, x, idx, segsize, step)
    # TODO bounds checks and warnings
    copyto!(buf, 1, x, idx, segsize)
    if step <= 0
        copyto!(x, idx + step + segsize, x, idx + step, abs(step))
    else    
        copyto!(x, idx, x, idx + segsize, step)
    end
    copyto!(x, idx + step, buf, 1, segsize)

    return x
end
moveto!(buf, x, srcidx, dstidx, segsize) = shift!(buf, x, srcidx, segsize, dstidx - srcidx)

function node_by_node!(cost, Ginfo, Oinfo, k, a, b, c) # TODO clean this up
    (; A) = Ginfo
    (; idx_to_embedding, position_to_idx, volume) = Oinfo

    n = length(idx_to_embedding)
    buf = zeros(1)
    best = evalorder(cost, A, idx_to_embedding)
    backup = deepcopy(Oinfo)

    for col in axes(A, 2)
        srcidx = col
        for dstidx in max(1, col - k):min(n, col + k)
            # println("Col $col Src $srcidx Dst $dstidx Eval $(evalorder(cost, A, idx_to_embedding))")
            # println(position_to_idx[max(1, col-k):min(n, col+k)])
            moveto!(buf, position_to_idx, srcidx, dstidx, 1)
            srcidx = dstidx
            inflate_embedding!(idx_to_embedding, position_to_idx, volume)
            inflate_embedding!(Oinfo)
            test = evalorder(cost, A, idx_to_embedding)
            if test <= best
                best = test
                copyorder!(backup, Oinfo)
            end
        end
    end
    copyorder!(Oinfo, backup)
    return Oinfo
end

