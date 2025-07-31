using Revise
using TullockDynamics
using Plots  # Required for plotting functionality

cost(x) = x
# cost(x) = 0.8x^1.001

function generate_Bayesian_agent_contest(n::Int, T::Int; χ, caching=:none)
    agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T; caching=caching)
end

# Set parameters
n = 3  # number of agents
T = 3000
χ = 0.05
contest = generate_Bayesian_agent_contest(n, T; χ = χ, caching=:exact);

ε=0.002
@time run!(contest, ε=ε, showprogress=true, accuracy=:strict)

@time run!(contest, ε=ε, showprogress=true, accuracy=:relaxed)

@time run!(contest, ε=ε, showprogress=true, accuracy=:veryrelaxed)

nash_gap(contest, T)

# final_efforts(contest)

visualise(contest, ylims=(0,0.3))