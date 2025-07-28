using Revise
using TullockDynamics

cost(x) = x
# cost(x) = 0.8x^1.001

function generate_Bayesian_agent_contest(n::Int, T::Int; χ)
    agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 8  # number of agents
T = 1000
χ = 0.05
contest = generate_Bayesian_agent_contest(n, T; χ = χ);

@profview_allocs run!(contest)

# final_efforts(contest)

# visualise(contest, ylims=(0,0.3))