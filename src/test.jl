using Ordering
using Graphs
using Coarsening
using Random: Xoshiro, randperm
# using MatrixDepot

tg = random_regular_graph(10000, 3) # SimpleGraph(matrixdepot("Pajek/Erdos971"))
tg, _ = induced_subgraph(tg, connected_components(tg)[1])
cost = Ordering.CrossingNumber() # Ordering.SumStrength(Ordering.SqChordLength())
algdist = Ordering.AlgebraicDistance(
    Ordering.gs_factory(Ordering.CircularRelaxation(0.5), Xoshiro(10)), 
    Ordering.SqChordLength(), 
    5
)
coarsening = VolumeCoarsening(0.4, 2.0)
W = 1.0 * adjacency_matrix(tg)
begin
    global accumulator = 100000000000000
    for seed in 1:10
        idx2pos = Ordering.ordergraph(cost, tg; coarsening=coarsening, order=2, seed=seed, coarsest=12, algdist=algdist, maxswaps=10)
        global accumulator = min(Ordering.evalorder(cost, W, idx2pos), accumulator)
    end
    println(accumulator)
end



W = 1.0 * adjacency_matrix(tg)

rb = Ordering.RecursiveBisection(0.03, "strong", ".", "/home/cameron/Work/KaHIP/", 5)
rbord = Ordering.ordermatrix(rb, W)
println(Ordering.evalorder(cost, W, rbord))

for i in 1:10
    Ordering.greedyswaps!(rbord, W, 10)
end
println(Ordering.evalorder(cost, W, rbord))