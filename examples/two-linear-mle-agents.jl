using TullockDynamics

linear_cost(x) = x
δ = 0.01

agents = [DetMLEAgent(linear_cost; δ=δ), DetMLEAgent(linear_cost; δ=δ)]

# initial_efforts = [0.1, 0.2]
initial_efforts = [rand(), rand()]

T = 10000

contest = TullockContest(agents, initial_efforts, T)

@profview run!(contest)

final_efforts(contest)

plot(contest, ylims=(0,0.5))


