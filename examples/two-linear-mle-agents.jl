using TullockDynamics
using StableRNGs

rng = StableRNG(123456789)

linear_cost(x) = x
χ = 0.01

agents = [MLEAgent(linear_cost; χ=χ), MLEAgent(linear_cost; χ=χ)]

initial_efforts = [rand(rng), rand(rng)]

T = 10

contest = TullockContest(agents, initial_efforts, T)

run!(contest, rng=rng)

# final_efforts(contest)

visualise(contest)


# using StatsBase
# rng = StableRNG(123456789)
# sample(rng, Weights([1,5,2]))