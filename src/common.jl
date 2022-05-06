
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
    level_id::Int
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
    (; level_id, Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; position_to_idx, idx_to_embedding, idx_to_position, volume) = Oinfo

    dropzeros!(A) # TODO necessary?
    @debug debug_level_info(level_id, Ginfo, Oinfo)

    strength = A # TODO compute connection strength
    Ac, P, Ginfo.C, Ginfo.F = coarsen(ord.config.coarsening, A; volume = volume, strength = strength, idx_to_position=idx_to_position) # TODO more generic
    Coarsening.fix_adjacency!(Ac)
    dropzeros!(Ac)

    cGinfo = GraphInfo(Ac, sumcols(Ac), nothing, nothing)
    cOinfo = init_coarse_order(Ginfo.C, P, Oinfo)

    push!(level, OrderLevel(level_id + 1, cGinfo, cOinfo))
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
    (; level_id, Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; idx_to_embedding) = Oinfo

    best = evalorder(ord.cost, A, idx_to_embedding)
    backup = deepcopy(Oinfo)
    preeval = best
    for perm in permutations(1:length(idx_to_embedding))
        idx_to_embedding .= perm
        inflate_embedding!(Oinfo) # TODO see if it's necessary to do the full process
        cur = evalorder(ord.cost, A, idx_to_embedding)
        if cur <= best
            copyorder!(backup, Oinfo)
            best = cur
        end
    end
    copyorder!(Oinfo, backup)
    posteval = best

    @debug debug_level_info(level_id, Ginfo, Oinfo)

    @debug """
    ########## Initial Level $level_id 
    Pre: $preeval
    Post: $posteval
    """

    return nothing
end


