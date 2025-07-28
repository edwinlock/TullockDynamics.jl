using TullockDynamics

linear_cost(x) = x

function generate_MLE_agent_contest(n::Int, T::Int; χ)
    agents = [MLEAgent(linear_cost; χ=χ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 10 # number of agents
T = 5000
χ = 0.05
contest = generate_MLE_agent_contest(n, T; χ = χ)
@time run!(contest)

# final_efforts(contest)

# visualise(contest)