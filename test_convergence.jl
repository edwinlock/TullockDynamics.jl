using TullockDynamics

cost(x) = x
agents = [BayesianAgent(cost; χ=0.05) for _ in 1:3]
initial_efforts = [0.1, 0.15, 0.2]
contest = TullockContest(agents, initial_efforts, 100)
run!(contest)

println("Final efforts: ", final_efforts(contest))
println("Final Nash gap: ", nash_gap(contest, min(100, size(contest.efforts, 2))))