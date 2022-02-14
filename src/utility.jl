function invperm!(dst, src)
    Base.require_one_based_indexing(src)
    Base.require_one_based_indexing(dst)
    n = length(src)
    length(dst) == n || throw(ArgumentError("argument dst is not the right size"))
    @inbounds for (i, j) in enumerate(a)
        ((1 <= j <= n) && dst[j] == 0) ||
            throw(ArgumentError("argument src is not a permutation"))
        dst[j] = i
    end
    return dst
end
