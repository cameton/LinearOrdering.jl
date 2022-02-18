
struct PSum <: AbstractOrderCost
    p::Int
end

function evalorder(ps::PSum, A, order)
    rows = rowvals(A)
    vals = nonzeros(A)
    acc = zero(eltype(A))
    for c in axes(A, 2)
        for idx in nzrange(A, c)
            row = rows[idx]
            if c < row
                acc += vals[idx] * abs(order[c] - order[row]) ^ ps.p
            end
        end
    end
    return acc
end

function smoothing(cost::PSum, A, d, x, order, volume, cols, rows) # TODO more general
    if cost.p == 2
        twosum_smoothing(A, d, x, cols, rows)
    elseif cost.p == 1
        onesum_smoothing(A, x, cols, rows)
    end
    embedding_to_order!(order, x)
    order_embedding!(x, order, volume)
    return nothing
end

function Multilevel.process_coarse!(ord::MLOrdering{PSum}, level)
    (; Ginfo, Oinfo) = first(level)
    (; A, d, C, F) = Ginfo
    (; volume, order, embedding) = Oinfo

    backup = deepcopy(embedding)
    backup_order = deepcopy(order)

    for _ in 1:ord.config.compat_sweeps
        backup .= embedding
        backup_order .= order
        smoothing(ord.cost, A, d, embedding, order, volume, :, F)
        if evalorder(ord.cost, A, backup) < evalorder(ord.cost, A, embedding)
            embedding .= backup
            order .= backup_order
            break
        end
    end

    val = evalorder(ord.cost, A, embedding)
    println("Size $(size(A, 1)) Post Compat $val")

    for _ in 1:ord.config.gauss_sweeps
        backup .= embedding
        backup_order .= order
        smoothing(ord.cost, A, d, embedding, order, volume, :, :)
        if evalorder(ord.cost, A, backup) < evalorder(ord.cost, A, embedding)
            embedding .= backup
            order .= backup_order
            break
        end
    end

    val = evalorder(ord.cost, A, embedding)
    println("Size $(size(A, 1)) Post Gauss $val")

    if ord.cost.p == 2
        # for windowsize in ord.config.windowsizes
        #       window_minimization(ord.cost, Ginfo, Oinfo; windowsize=windowsize, config=ord.config)
        #       window_minimization(ord.cost, Ginfo, Oinfo; windowsize=windowsize, config=ord.config, rev=true)
        # end
        node_by_node!(ord.cost, A, embedding, order, volume, ord.config.node_window_size)
    elseif ord.cost.p == 1
        node_by_node!(ord.cost, A, embedding, order, volume, ord.config.node_window_size)
    end

    val = evalorder(ord.cost, A, embedding)
    println("Size $(size(A, 1)) Post Strict $val")
    embedding_to_order!(order, embedding)
    order_embedding!(embedding, order, volume)

    println("Embedding $(embedding[1:5])")
    println("Order $((order)[1:5])")

    val = evalorder(ord.cost, A, invperm(order))
    println("Size $(size(A, 1)) Post Strict $val")
    return nothing
end 

function sumcols!(b, A, cols)
    for col in cols
        Coarsening.addcol!(b, A, col)
    end
    return b
end

function Multilevel.uncoarsen!(ord::MLOrdering{PSum}, level)
    (; Oinfo) = pop!(level)
    embeddingc = Oinfo.embedding
    (; Ginfo, Oinfo) = first(level)
    (; A, C, F) = Ginfo
    (; order, embedding, volume) = Oinfo

    embedding .= 0
    embedding[C] = embeddingc

    connection = sumcols(A, C)
    smoothing(ord.cost, A, connection, embedding, order, volume, C, F)

    val = evalorder(ord.cost, A, embedding)
    println("Size $(size(A, 1)) Post Interpolation $val")


    randorder = randperm(size(A, 1))
    randembed = zeros(size(A, 1))
    order_embedding!(randembed, randorder, volume)
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
    order = randperm(Xoshiro(config.seed), nv(G))
    embedding = similar(order, Float64)
    volume = ones(nv(G))
    order_embedding!(embedding, order, volume)
    println("STARTING COST $(evalorder(cost, A, embedding))")
    top = OrderLevel{Float64}(GraphInfo(A, sumcols(A), nothing, nothing), OrderInfo(order, embedding, volume))
    level = Stack{OrderLevel{Float64}}()
    push!(level, top)
    Multilevel.cycle!(ord, level)
    return order
end

