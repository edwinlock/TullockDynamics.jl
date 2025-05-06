using TullockDynamics
import Random
Random.seed!(123456789)

linear_cost(x) = x
χ = 0.01

agents = [MLEAgent(linear_cost; χ=χ), MLEAgent(linear_cost; χ=χ)]

initial_efforts = [rand(), rand()]

T = 10

contest = TullockContest(agents, initial_efforts, T)

run!(contest)

println(final_efforts(contest))

visualise(contest)

