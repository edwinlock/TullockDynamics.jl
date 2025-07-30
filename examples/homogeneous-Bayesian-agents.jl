using Revise
using TullockDynamics
using Plots  # Required for plotting functionality

cost(x) = x
# cost(x) = 0.8x^1.001

function generate_Bayesian_agent_contest(n::Int, T::Int; χ, accuracy::Symbol = :relaxed)
    agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T; accuracy=accuracy)
end

# Set parameters
n = 5  # number of agents
T = 1000
χ = 0.05
contest = generate_Bayesian_agent_contest(n, T; χ = χ, accuracy = :relaxed);  # Uses default :relaxed accuracy with caching

@time run!(contest)

nash_gap(contest, T)

# final_efforts(contest)

visualise(contest, ylims=(0,0.3))