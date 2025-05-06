using Revise
using TullockDynamics

# cost(x) = x
cost(x) = 0.8x^1.001

function generate_Bayesian_agent_contest(n::Int, T::Int; χ)
    agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 3  # number of agents
T = 100
χ = 0.05
contest = generate_Bayesian_agent_contest(n, T; χ = χ)
run!(contest)

final_efforts(contest)
visualise(contest, ylims=(0,0.3))

# # Debugging code
# # We want to look at the estimator PDFs of agent i in time step t
# using Plots
# min_total_efforts = sum(agent.χ for agent in contest.agents)
# max_total_efforts = sum(agent.max_effort for agent in contest.agents)

# function plot_estimator(i, t, contest)
#     agent = contest.agents[i]
#     mem_window = 1:t
#     own_efforts, total_efforts, wins = retrieve_estimator_data(i, mem_window, contest)
#     est = agent.estimator(
#         own_efforts=own_efforts,
#         total_efforts=total_efforts,
#         wins=wins,
#         min_other_efforts=min_total_efforts - agent.χ,
#         max_other_efforts=max_total_efforts - agent.max_effort,
#     )
#     display(plot!(est, 0, max_total_efforts+1))
# end

# plot()
# plot_estimator(1, 1, contest)
# plot_estimator(1, 2, contest)
# plot_estimator(1, 3, contest)
# plot_estimator(1, 4, contest)
# plot_estimator(1, 5, contest)
