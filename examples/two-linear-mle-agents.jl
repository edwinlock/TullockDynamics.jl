using TullockDynamics
import Random
Random.seed!(123456789)

rng = Random.default_rng()

linear_cost(x) = x
χ = 0.01

agents = [MLEAgent(linear_cost; χ=χ), MLEAgent(linear_cost; χ=χ)]

initial_efforts = [rand(rng), rand(rng)]

T = 10

contest = TullockContest(agents, initial_efforts, T)

run!(contest, rng=rng)

println(final_efforts(contest))

visualise(contest)

