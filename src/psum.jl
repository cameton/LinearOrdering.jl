
struct PSum <: AbstractOrderCost
    p::Int
end

function evalorder(ps::PSum, A, idx_to_position)
    rows = rowvals(A)
    vals = nonzeros(A)
    acc = zero(eltype(A))
    for c in axes(A, 2)
        for idx in nzrange(A, c)
            row = rows[idx]
            if c < row
                acc += vals[idx] * abs(idx_to_position[c] - idx_to_position[row]) ^ ps.p
            end
        end
    end
    return acc
end

function smoothing(cost::PSum, Ginfo, Oinfo, cols, rows) # TODO more general
    (; A, d) = Ginfo
    (; position_to_idx, idx_to_embedding, volume) = Oinfo

    if cost.p == 2
        twosum_smoothing(A, d, idx_to_embedding, cols, rows)
    elseif cost.p == 1
        onesum_smoothing(A, idx_to_embedding, cols, rows)
    end
    inflate_embedding!(Oinfo)
    return nothing
end

function Multilevel.process_coarse!(ord::MLOrdering{PSum}, level)
    (; Ginfo, Oinfo) = first(level)
    (; A, d, C, F) = Ginfo
    (; volume, position_to_idx, idx_to_embedding) = Oinfo

    best = evalorder(ord.cost, A, idx_to_embedding)
    backup = deepcopy(Oinfo)

    for _ in 1:ord.config.compat_sweeps
        smoothing(ord.cost, Ginfo, Oinfo, :, F)
        test = evalorder(ord.cost, A, idx_to_embedding)
        if test <= best
            copyorder!(backup, Oinfo)
            best = test
        else
            break
        end
    end
    copyorder!(Oinfo, backup)

    val = evalorder(ord.cost, A, idx_to_embedding)
    println("Size $(size(A, 1)) Post Compat $val")

    for _ in 1:ord.config.gauss_sweeps
        smoothing(ord.cost, Ginfo, Oinfo, :, :)
        test = evalorder(ord.cost, A, idx_to_embedding)
        if test <= best
            copyorder!(backup, Oinfo)
            best = test
        else
            break
        end
    end
    copyorder!(Oinfo, backup)

    val = evalorder(ord.cost, A, idx_to_embedding)
    println("Size $(size(A, 1)) Post Gauss $val")

    if ord.cost.p == 2
        # for windowsize in ord.config.windowsizes
        #       window_minimization(ord.cost, Ginfo, Oinfo; windowsize=windowsize, config=ord.config)
        #       window_minimization(ord.cost, Ginfo, Oinfo; windowsize=windowsize, config=ord.config, rev=true)
        # end
        node_by_node!(ord.cost, Ginfo, Oinfo, ord.config.node_window_size)
    elseif ord.cost.p == 1
        node_by_node!(ord.cost, Ginfo, Oinfo, ord.config.node_window_size)
    end

    val = evalorder(ord.cost, A, idx_to_embedding)
    println("Size $(size(A, 1)) Post Strict $val")

    return nothing
end 

function Multilevel.uncoarsen!(ord::MLOrdering{PSum}, level)
    (; Oinfo) = pop!(level)
    coarse_idx_to_embedding = Oinfo.idx_to_embedding

    (; Ginfo, Oinfo) = first(level)
    (; A, C, F) = Ginfo
    (; position_to_idx, idx_to_embedding, volume) = Oinfo

    idx_to_embedding .= 0 # TODO should it be like this
    idx_to_embedding[C] = coarse_idx_to_embedding

    connection = sumcols(A, C) # TODO check if this is right
    tmp_thing = GraphInfo(Ginfo.A, connection, Ginfo.C, Ginfo.F) # TODO this is janky, fix it
    smoothing(ord.cost, tmp_thing, Oinfo, C, F)

    val = evalorder(ord.cost, A, idx_to_embedding)
    println("Size $(size(A, 1)) Post Interpolation $val")


    randorder = randperm(size(A, 1))
    randembed = zeros(size(A, 1))
    inflate_embedding!(randembed, randorder, volume)
    randobj = evalorder(ord.cost, A, randembed)
    println("RANDORDER $randobj Ratio $(val / randobj)")
   
end

sumcols(A, cols=1:size(A, 2)) = sumcols!(zeros(size(A, 1)), A, cols)

using Random

# windowsizes, compat_sweeps, window_sweeps, stride, gauss_sweeps, coarsening, pad_percent
function ordergraph(cost, G; config...)
    config = NamedTuple(config)
    ord = MLOrdering(cost, config)
    A = adjacency_matrix(G) * 1.0
    idx_to_position = randperm(Xoshiro(config.seed), nv(G))
    position_to_idx = invperm(idx_to_position)
    idx_to_embedding = collect(Float64, idx_to_position)
    volume = ones(nv(G))
    Oinfo = OrderInfo(idx_to_embedding, position_to_idx, idx_to_position, volume)
    println("STARTING COST $(evalorder(cost, A, idx_to_embedding))")
    top = OrderLevel{Float64}(GraphInfo(A, sumcols(A), nothing, nothing), Oinfo)
    level = Stack{OrderLevel{Float64}}()
    push!(level, top)
    Multilevel.cycle!(ord, level)
    return position_to_idx, idx_to_position
end

