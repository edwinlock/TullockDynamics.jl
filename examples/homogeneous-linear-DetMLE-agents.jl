using TullockDynamics

linear_cost(x) = x

function generate_det_MLE_agent_contest(n::Int, T::Int; δ)
    agents = [DetMLEAgent(linear_cost; δ=δ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 10  # number of agents
T = 10000
δ = 0.05
contest = generate_det_MLE_agent_contest(n, T; δ = δ)
run!(contest)

final_efforts(contest)

plot(contest, ylims=(0,0.3))

