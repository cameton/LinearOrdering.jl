
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

function smoothing(cost::PSum, A, x, order, volume, cols) # TODO more general
    twosum_smoothing(A, x, cols)
    order .= sortperm(x)
    order_embedding!(x, order, volume)
    return nothing
end

function Multilevel.process_coarse!(ord::MLOrdering{PSum}, level) # TODO figure out the F vs Fc situation
    (; A, volume, order, embedding) = first(level)
    for _ in ord.config.gauss_sweeps
        smoothing(ord.cost, A, embedding, order, volume, order)
    end
    for windowsize in ord.config.windowsizes
        window_minimization(ord.cost, A, order, embedding, volume; windowsize=windowsize, config=ord.config)
        window_minimization(ord.cost, A, order, embedding, volume; windowsize=windowsize, config=ord.config, rev=true)
    end

end 

function sumcols!(b, A, cols)
    for col in cols
        Coarsening.addcol!(b, A, col)
    end
    return b
end

function Multilevel.uncoarsen!(ord::MLOrdering{PSum}, level)
    (; C, F, embedding) = pop!(level)
    Cc, Fc, embeddingc = C, F, embedding
    (; A, volume, order, embedding) = first(level)
    embedding .= 0
    embedding[Cc] = embeddingc

    connection = sumcols!(zeros(size(order, 1)), A, Cc)
    for idx in sortperm(Fc; by = f -> connection[f], rev=true) # TODO adapt order as points are added
        twosum_smoothing(A, embedding, idx) # TODO more general, own function, more performant
    end
    order .= sortperm(embedding)
    order_embedding!(embedding, order, volume)

    for _ in ord.config.compat_sweeps
        smoothing(ord.cost, A, embedding, order, volume, Fc)
    end
end

# windowsizes, compat_sweeps, window_sweeps, stride, gauss_sweeps, coarsening, padding
function ordergraph(cost, G; config...)
    ord = MLOrdering(cost, NamedTuple(config))
    A = adjacency_matrix(G) * 1.0
    order = collect(1:nv(G))
    embedding = zeros(nv(G))
    volume = ones(nv(G))
    order_embedding!(embedding, order, volume)
    top = OrderLevel{Float64}(A, volume, [], [], order, embedding)
    level = Stack{OrderLevel{Float64}}()
    push!(level, top)
    Multilevel.cycle!(ord, level)
    return order
end

