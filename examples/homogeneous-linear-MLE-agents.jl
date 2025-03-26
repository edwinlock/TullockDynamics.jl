using TullockDynamics

linear_cost(x) = x

function generate_MLE_agent_contest(n::Int, T::Int; δ)
    agents = [MLEAgent(linear_cost; δ=δ) for _ in 1:n]
    initial_efforts = [rand() for _ in 1:n]
    return TullockContest(agents, initial_efforts, T)
end

# Set parameters
n = 3 # number of agents
T = 100
δ = 0.05
contest = generate_MLE_agent_contest(n, T; δ = δ)
run!(contest)

final_efforts(contest)

plot(contest.efforts, ylims=(0,0.3))

plot(contest.utilities, ylims=(0,0.3))

plot(contest.nash_gaps, ylims=(0,0.3))

plot(t -> nash_gap(contest, t), 1:1000)