using TullockDynamics

linear_cost(x) = x

function generate_det_MLE_agent_contest(n::Int, T::Int; χ)
    agents = [DetMLEAgent(linear_cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 5  # number of agents
T = 10
χ = 0.05
contest = generate_det_MLE_agent_contest(n, T; χ = χ)
run!(contest)

final_efforts(contest)

visualise(contest, ylims=(0,0.3))