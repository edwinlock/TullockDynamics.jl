using TullockDynamics

cost(x) = x
println("=== Testing improved-caching branch ===")

# Test 1: Small contest
agents = [BayesianAgent(cost; χ=0.05) for _ in 1:3]
initial_efforts = [0.1, 0.15, 0.2]
contest = TullockContest(agents, initial_efforts, 100)
run!(contest)

println("Test 1 - Small contest:")
println("  Final efforts: ", final_efforts(contest))
println("  Final Nash gap: ", nash_gap(contest, min(100, size(contest.efforts, 2))))

# Test 2: Let's check what's in the estimator calls
# Print some debug info from the first few rounds
agent1 = agents[1]
println("\nTest 2 - Debug first few rounds:")
for t in 2:5
    mem_window = agent1.h(t)
    println("  Round $t, memory window: $mem_window")
    if !isempty(mem_window)
        estim = agent1.estimator(contest, 1, mem_window)
        println("  Estimator type: ", typeof(estim))
        if estim isa Function
            println("  Estimator is a PDF function")
            # Try to evaluate it at a few points
            try
                println("  PDF(0.1) = ", estim(0.1))
                println("  PDF(0.2) = ", estim(0.2))
            catch e
                println("  Error evaluating PDF: ", e)
            end
        else
            println("  Estimator value: ", estim)
        end
    end
end

# Test 3: Check workspace bounds
println("\nTest 3 - Workspace bounds:")
println("  min_other_bounds: ", contest.workspace.min_other_bounds)
println("  max_other_bounds: ", contest.workspace.max_other_bounds)