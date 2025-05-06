using Revise
using TullockDynamics

linear_cost(x) = x
# quadratic_cost(x) = x^2

function generate_Bayesian_agent_contest(n::Int, T::Int; χ)
    agents = [BayesianAgent(linear_cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 5  # number of agents
T = 5
χ = 0.05
contest = generate_Bayesian_agent_contest(n, T; χ = χ)
run!(contest)

final_efforts(contest)

visualise(contest, ylims=(0,0.3))