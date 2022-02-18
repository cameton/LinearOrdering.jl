
struct MLOrdering{C<:AbstractOrderCost, T}
    cost::C
    config::T
end

struct OrderInfo{T <: Real}
    order::Vector{Int}
    embedding::Vector{T}
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

lazy_cumsum(iter) = Iterators.accumulate(+, iter)

function order_embedding!(x, order, volume)
    for (i, v) in enumerate(lazy_cumsum(volume[idx] for idx in order))
        x[i] = v
    end
    invpermute!(x, order)
    x .-= volume ./ 2
    return x
end
function embedding_to_order!(order, x) # TODO write better
    # order .= invperm(sortperm(x))
    sortperm!(order, x)
    return order
end

function initorder(C, order, volume)
    orderc = sortperm(C; by=idx -> order[idx])
    embeddingc = similar(volume, length(C))
    order_embedding!(embeddingc, orderc, volume)
    return orderc, embeddingc
end


function Multilevel.coarsen!(ord::MLOrdering, level)
    (; Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; order, embedding, volume) = Oinfo
    dropzeros!(A)

    testG = SimpleGraph(Symmetric(A))
    println("Fine Vertices $(nv(testG)) Edges $(ne(testG)) MinDeg $(minimum(degree(testG))) MaxDeg $(maximum(degree(testG))) MeanDeg $(sum(degree(testG)) / nv(testG))")
    println("Fine Volume Max $(maximum(volume)) Min $(minimum(volume)) Mean $(sum(volume) / length(volume))")
    println("Fine Adj Max $(maximum(A)) Min $(minimum(nonzeros(A))) Mean $(sum(nonzeros(A)) / length(nonzeros(A)))")
    println("Fine total volume $(sum(volume))")
    strength = A # TODO compute connection strength
    Ac, P, Ginfo.C, Ginfo.F = coarsen(ord.config.coarsening, A; volume = volume, strength = strength) # TODO more generic
    Coarsening.fix_adjacency!(Ac)
    println("")
    dropzeros!(Ac)

    testG = SimpleGraph(Symmetric(Ac))
    println("Coarse Vertices $(nv(testG)) Edges $(ne(testG)) MinDeg $(minimum(degree(testG))) MaxDeg $(maximum(degree(testG))) MeanDeg $(sum(degree(testG)) / nv(testG))")
    volumec = P' * volume


    println("Coarse Volume Max $(maximum(volumec)) Min $(minimum(volumec)) Mean $(sum(volumec) / length(volumec))")

    println("Coarse Adj Max $(maximum(Ac)) Min $(minimum(nonzeros(Ac))) Mean $(sum(nonzeros(Ac)) / length(nonzeros(Ac)))")

    println("Coarse total volume $(sum(volumec))")
    println("")
    println("-----------------------------------")
    println("")

    nc = length(Ginfo.C)
    cGinfo = GraphInfo(Ac, sumcols(Ac), nothing, nothing)
    cOinfo = OrderInfo(initorder(Ginfo.C, order, volumec)..., volumec)

    push!(level, OrderLevel(cGinfo, cOinfo))
    return nothing
end


function Multilevel.doinitial(ord::MLOrdering, level)
    (; Oinfo) = first(level)
    return length(Oinfo.order) < ord.config.coarsest
end

function Multilevel.initial!(ord::MLOrdering, level)
    (; Ginfo, Oinfo) = first(level)
    (; A) = Ginfo
    (; order, embedding, volume) = Oinfo

    best = evalorder(ord.cost, A, embedding)
    println("Coarsest PreEval $best")
    for perm in permutations(order)
        order_embedding!(embedding, perm, volume)
        cur = evalorder(ord.cost, A, embedding)
        if cur < best
            order .= perm
            best = cur
        end
    end
    order_embedding!(embedding, order, volume)
    println("Coarsest PostEval $best")
    return nothing
end

function shift(order, col, dir)
    while dir != 0
        if dir < 0
            order[col], order[col - 1] = order[col - 1], order[col]
            col -= 1
            dir += 1
        else
            order[col], order[col + 1] = order[col + 1], order[col]
            col += 1
            dir -= 1
        end
    end
    return order
end

function node_by_node!(cost, A, embedding, order, volume, k)
    n = length(order)
    for col in axes(A, 2)
        best = evalorder(cost, A, embedding)
        candidate = deepcopy(embedding)
        lo = min(k, col - 1) 
        locol = col - lo
        winsize = lo + min(k, n - col) + 1
        shift(candidate, col, -lo)
        col = locol
        test = evalorder(cost, A, candidate) # TODO make less repetitive
        if test < best
            best = test
            copyto!(embedding, locol, candidate, locol, winsize)
        end
        for i in 1:(winsize - 1)
            shift(candidate, col, 1)
            col += 1
            test = evalorder(cost, A, candidate)
            if test < best
                best = test
                copyto!(embedding, locol, candidate, locol, winsize)
            end
        end
    end

    embedding_to_order!(order, embedding)
    order_embedding!(embedding, order, volume)
    return order
end

