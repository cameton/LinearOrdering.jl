
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

lazy_cumsum(iter) = Iterators.accumulate(+, iter)
lazy_cumdiff(v, total) = Iterators.accumulate(-, v, init=total)

function sumcols!(b, A, cols)
    for col in cols
        Coarsening.addcol!(b, A, col)
    end
    return b
end

print_graph_info(label, testG) = println("$label Vertices $(nv(testG)) Edges $(ne(testG)) MinDeg $(minimum(degree(testG))) MaxDeg $(maximum(degree(testG))) MeanDeg $(sum(degree(testG)) / nv(testG))")
print_volume_info(label, volume) = println("$label Volume Total $(sum(volume)) Max $(maximum(volume)) Min $(minimum(volume)) Mean $(sum(volume) / length(volume))") 
print_adj_info(label, A) = println("$label Adj Max $(maximum(A)) Min $(minimum(nonzeros(A))) Mean $(sum(nonzeros(A)) / length(nonzeros(A)))")

