using StatsBase

"""
Workspace for intermediate computations to avoid repeated allocations.
"""
mutable struct ContestWorkspace
    # Pre-allocated buffers for repeated use
    all_efforts::Vector{Float64}        # Buffer for contest.efforts[:,t]
    weights_obj::Any  # Reusable Weights object (type will be inferred)
    
    # Pre-computed constants (computed once in constructor)
    min_total_efforts::Float64          # sum(agent.χ for agent in agents)
    max_total_efforts::Float64          # sum(agent.max_effort for agent in agents)
    
    # Per-agent pre-computed bounds (avoid repeated calculations)
    min_other_bounds::Vector{Float64}   # max(0.001, min_total_efforts - agent.max_effort)
    max_other_bounds::Vector{Float64}   # max(0.01, max_total_efforts - agent.χ)
    
    # Reusable variables  
    total_effort::Float64               # sum(all_efforts) - computed once per round
end

struct TullockContest
    agents::Vector{Agent}
    efforts::Matrix{Float64}
    winners::Matrix{Bool}
    utilities::Matrix{Float64}
    nash_gaps::Matrix{Float64}
    workspace::ContestWorkspace         # Workspace for optimizations
end

num_rounds(contest::TullockContest) = size(contest.efforts)[2]


"""Create Tullock Contest with given `agents`, initial effort vector `x`, and `T` rounds."""
function TullockContest(agents::Vector{Agent}, x::Vector{Float64}, T::Int)
    @assert length(agents) >= 2 "Contest must have at least 2 agents."
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
    # Pre-compute constants for workspace
    min_total_efforts = sum(agent.χ for agent in agents)
    max_total_efforts = sum(agent.max_effort for agent in agents)
    
    # Pre-compute per-agent bounds to avoid repeated calculations
    min_other_bounds = [max(0.001, min_total_efforts - agent.max_effort) for agent in agents]
    max_other_bounds = [max(0.01, max_total_efforts - agent.χ) for agent in agents]
    
    # Initialize workspace
    all_efforts_buffer = zeros(num_agents)
    weights_obj = StatsBase.Weights(all_efforts_buffer)  # Create reusable Weights object
    workspace = ContestWorkspace(
        all_efforts_buffer,     # all_efforts buffer
        weights_obj,            # reusable Weights object
        min_total_efforts,      # pre-computed constant
        max_total_efforts,      # pre-computed constant
        min_other_bounds,       # pre-computed per-agent bounds
        max_other_bounds,       # pre-computed per-agent bounds
        0.0                     # total_effort (will be computed each round)
    )
    
    return TullockContest(agents, efforts, winners, utilities, nash_gaps, workspace)
end


"""
Compute the Nash gap of contest in round `t`.
"""
function nash_gap(contest::TullockContest, t::Int)
    return sum(contest.nash_gaps[:,t])
end