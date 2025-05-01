using TullockDynamics

linear_cost(x) = x
χ = 0.01

agents = [MLEAgent(linear_cost; χ=χ), MLEAgent(linear_cost; χ=χ)]

# initial_efforts = [0.1, 0.2]
initial_efforts = [rand(), rand()]

T = 10000

contest = TullockContest(agents, initial_efforts, T)

run!(contest)

final_efforts(contest)

visualise(contest, ylims=(0,0.5))
