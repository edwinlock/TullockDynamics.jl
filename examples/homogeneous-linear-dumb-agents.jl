using TullockDynamics
using StableRNGs

rng = StableRNG(123456789)

linear_cost(x) = x

function generate_dumb_agent_contest(n::Int, T::Int; χ)
    agents = [DumbAgent(linear_cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 2  # number of agents
T = 1000
χ = 0.05
contest = generate_MLE_agent_contest(n, T; χ = χ)
run!(contest, rng=rng)

final_efforts(contest)

plot(contest, ylims=(0,0.3))

