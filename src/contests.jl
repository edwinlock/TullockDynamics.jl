struct TullockContest
    agents::Vector{Agent}
    efforts::Matrix{Float64}
    winners::Matrix{Bool}
    utilities::Matrix{Float64}
    nash_gaps::Matrix{Float64}
end

num_rounds(contest::TullockContest) = size(contest.efforts)[2]


"""Create Tullock Contest with given `agents`, initial effort vector `x`, and `T` rounds."""
function TullockContest(agents::Vector{Agent}, x::Vector{Float64}, T::Int)
    @assert length(agents) == length(x) "Length of effort vector must match number of agents."
    @assert T ≥ 1 "Must have positive number of rounds."
    num_agents = length(agents)
    # Create matrices
    efforts = zeros(num_agents, T)
    winners = falses(num_agents, T)
    utilities = zeros(num_agents, T)
    nash_gaps = zeros(num_agents, T)
    # Set initial efforts
    efforts[:,1] .= x
    return TullockContest(agents, efforts, winners, utilities, nash_gaps)
end


"""
Compute the Nash gap of contest in round `t`.
"""
function nash_gap(contest::TullockContest, t::Int)
    return sum(contest.nash_gaps[:,t])
end