using TullockDynamics

cost(x) = x
println("=== Testing main branch ===")

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
        min_total_efforts = sum(agent.χ for agent in contest.agents)
        max_total_efforts = sum(agent.max_effort for agent in contest.agents)
        own_efforts, total_efforts, wins = retrieve_estimator_data(1, mem_window, contest)
        
        estim = agent1.estimator(
            own_efforts=own_efforts,
            total_efforts=total_efforts,
            wins=wins,
            cost=agent1.cost,
            min_other_efforts=min_total_efforts - agent1.χ,
            max_other_efforts=max_total_efforts - agent1.max_effort,
        )
        
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

# Test 3: Check bounds computation
println("\nTest 3 - Bounds computation (main branch):")
min_total_efforts = sum(agent.χ for agent in contest.agents)
max_total_efforts = sum(agent.max_effort for agent in contest.agents)
println("  min_total_efforts: ", min_total_efforts)
println("  max_total_efforts: ", max_total_efforts)
println("  min_other_efforts for agent 1: ", min_total_efforts - agents[1].χ)
println("  max_other_efforts for agent 1: ", max_total_efforts - agents[1].max_effort)