
function invperm!(dst, src)
    Base.require_one_based_indexing(src)
    Base.require_one_based_indexing(dst)
    n = length(src)
    fill!(dst, 0)
    length(dst) == n || throw(ArgumentError("argument dst is not the right size"))
    @inbounds for (i, j) in enumerate(src)
        ((1 <= j <= n) && dst[j] == 0) ||
            throw(ArgumentError("argument src is not a permutation"))
        dst[j] = i
    end
    return dst
end

lazy_cumsum(iter; init=0) = Iterators.accumulate(+, iter, init=init)
lazy_cumdiff(v; init=sum(v)) = Iterators.accumulate(-, v, init=init)

function sumcols!(b, A, cols)
    for col in cols
        Coarsening.addcol!(b, A, col)
    end
    return b
end

function debug_level_info(level_id, Ginfo, Oinfo)
    (; A) = Ginfo
    (; volume) = Oinfo
    G = SimpleGraph(Symmetric(A)) # TODO ensure symmetric elsewhere - do something more efficient
    degreeG = degree(G)
    nzA = nonzeros(A)
    return """ 
    ########### $level_id
    Vertices $(nv(G)) Edges $(ne(G))
    MinDeg $(minimum(degreeG)) MaxDeg $(maximum(degreeG)) MeanDeg $(mean(degreeG))
    TotalVol $(sum(volume)) MaxVol $(maximum(volume)) MinVol $(minimum(volume)) MeanVol $(mean(volume))
    MaxAdj $(maximum(nzA)) MinAdj $(minimum(nzA)) MeanAdj $(mean(nzA))
    ###########
    """
end

