using TullockDynamics
using Plots  # Required for plotting functionality
import Random
Random.seed!(123456789)

linear_cost(x) = x
χ = 0.01

agents = [MLEAgent(linear_cost; χ=χ), MLEAgent(linear_cost; χ=χ)]

initial_efforts = [rand(), rand()]

T = 100

contest = TullockContest(agents, initial_efforts, T)

run!(contest)

println(final_efforts(contest))

visualise(contest)

