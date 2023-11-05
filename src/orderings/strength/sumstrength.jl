
struct SumStrength{T<:AbstractStrength}
    strength::T
end

minimize_position(cost::SumStrength, W, emb, v) = minimize_position(cost.strength, W, emb, v)

# compatible!(ord.config.refinement, info_fine.embedding, info_coarse.seeds) do (x, v)
#     return minimize_position(ord.cost, info_fine.W, x, v)
# end
# sortperm!(info_fine.idx2pos, info_fine.embedding)
# refine!(ord.config.refinement, info_fine.embedding) do x, v
#     return minimize_position(ord.cost, info_fine.W, x, v)
# end
# sortperm!(info_fine.idx2pos, info_fine.embedding)