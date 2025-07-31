using StatsBase

"""  
    TullockContest

A Tullock contest simulation tracking multiple agents competing through effort allocation.

This immutable struct contains the complete state of a contest including agent behaviors,
historical data, and an optimized workspace for efficient computation.

# Fields
- `agents::Vector{Agent}`: Competing agents with their strategies and cost functions
- `efforts::Matrix{Float64}`: Agent effort levels over time (agents × rounds)
- `winners::Matrix{Bool}`: Boolean indicators of round winners (agents × rounds)
- `utilities::Matrix{Float64}`: Agent utilities over time (agents × rounds)
- `nash_gaps::Matrix{Float64}`: Distance from Nash equilibrium (agents × rounds)
- `workspace::ContestWorkspace`: Pre-allocated buffers and configuration for performance

# Contest Mechanics
In each round:
1. Agents probabilistically decide whether to update their effort
2. If updating, they estimate opponents' total effort using their learning algorithm
3. They compute their best response to this estimate
4. A winner is selected probabilistically based on effort shares
5. The process repeats until convergence or maximum rounds

# Performance Design
- Pre-allocated matrices eliminate memory allocations during simulation
- Workspace contains pre-computed constants and reusable buffers
- Optimized for cache-friendly memory access patterns
"""
struct TullockContest
    agents::Vector{Agent}
    efforts::Matrix{Float64}
    winners::Matrix{Bool}
    utilities::Matrix{Float64}
    nash_gaps::Matrix{Float64}
    workspace::ContestWorkspace
end

"""  
    num_rounds(contest::TullockContest) -> Int

Return the total number of rounds in the contest.

This is determined by the second dimension of the efforts matrix, which is pre-allocated
during contest construction.
"""
num_rounds(contest::TullockContest) = size(contest.efforts)[2]


"""
Convert accuracy symbol to (atol, reltol) tolerance values for numerical integration.
"""
function accuracy_to_tolerances(accuracy::Symbol)
    if accuracy == :relaxed
        return 1e-8, 1e-6  # More stringent: was 1e-6, 1e-4
    elseif accuracy == :veryrelaxed  
        return 1e-5, 1e-3
    elseif accuracy == :strict
        return 1e-14, 1e-12  # Ultra high precision for maximum convergence accuracy
    else  # :default
        return 1e-10, 1e-8
    end
end

"""
    TullockContest(agents::Vector{Agent}, x::Vector{Float64}, T::Int; accuracy::Symbol = :relaxed, caching::Symbol = :approximate, cache_tolerance::Float64 = 1e-9) -> TullockContest

Create a Tullock contest simulation with specified agents, initial efforts, and time horizon.

A Tullock contest is a strategic game where agents compete by exerting costly effort, with the 
probability of winning proportional to their relative effort. This constructor sets up the 
contest data structure with optimized workspace for efficient simulation.

# Arguments
- `agents::Vector{Agent}`: Vector of competing agents (minimum 2 required)
- `x::Vector{Float64}`: Initial effort levels for each agent (must match length of agents)
- `T::Int`: Number of rounds to simulate (must be positive)
- `accuracy::Symbol = :relaxed`: Integration tolerance level for numerical methods
  - `:strict`: Ultra-high precision (atol=1e-14, reltol=1e-12) - slowest but maximum accuracy
  - `:default`: High precision (atol=1e-10, reltol=1e-8) - very accurate
  - `:relaxed`: Balanced precision (atol=1e-8, reltol=1e-6) - default, good speed/accuracy balance
  - `:veryrelaxed`: Lower precision (atol=1e-5, reltol=1e-3) - fastest with small accuracy loss
- `caching::Symbol = :approximate`: Root finding caching strategy for performance optimization
  - `:none`: No caching (uses original find_root)
  - `:exact`: Exact caching (cache hits only for identical x values)
  - `:approximate`: Approximate caching (default, cache hits for nearby x values within tolerance) 
- `cache_tolerance::Float64 = 1e-9`: Tolerance for approximate caching (only used when caching=:approximate)

# Returns
- `TullockContest`: Initialized contest ready for simulation

# Contest Structure
The contest maintains several matrices tracking the dynamics:
- **efforts**: Agent effort levels over time (agents × rounds)
- **winners**: Boolean matrix indicating round winners (agents × rounds)  
- **utilities**: Agent utilities over time (agents × rounds)
- **nash_gaps**: Distance from Nash equilibrium (agents × rounds)
- **workspace**: Optimized memory workspace for computations

# Validation
- Requires at least 2 agents (single-agent contests not meaningful)
- Initial effort vector length must match number of agents
- Number of rounds must be positive
- All initial efforts should be non-negative

# Performance Features
- **Workspace optimization**: Pre-allocated buffers eliminate memory allocations
- **Pre-computed constants**: Agent bounds calculated once for efficiency
- **Optimized data structures**: Efficient matrix layouts for cache performance
- **Configurable integration tolerances**: Adjustable precision for numerical integration (affects Bayesian agents only)

# Example
```julia
# Create agents with different cost functions
cost1(x) = x^2
cost2(x) = 0.5 * x^1.5
agents = [MLEAgent(cost1), DetMLEAgent(cost2)]

# Set up contest with initial efforts and 50 rounds (uses default :relaxed accuracy and approximate caching)
initial_efforts = [0.1, 0.15]
contest = TullockContest(agents, initial_efforts, 50)

# For highest precision, use default accuracy
bayesian_agents = [BayesianAgent(cost1), BayesianAgent(cost2)]
precise_contest = TullockContest(bayesian_agents, initial_efforts, 50; accuracy=:default)

# For maximum speed, use veryrelaxed accuracy
fast_contest = TullockContest(bayesian_agents, initial_efforts, 50; accuracy=:veryrelaxed)

# Disable caching if needed for testing or comparison
no_cache_contest = TullockContest(agents, initial_efforts, 50; caching=:none)

# Run simulation
final_round = run!(contest)
```

# See Also
- [`run!`](@ref): Execute the contest simulation
- [`Agent`](@ref): Agent constructor for custom agents
- [`visualise`](@ref): Plot contest dynamics (requires Plots.jl)
"""
function TullockContest(agents::Vector{Agent}, x::Vector{Float64}, T::Int; caching::Symbol = :approximate, cache_tolerance::Float64 = 1e-9)
    @assert length(agents) >= 2 "Contest must have at least 2 agents."
    @assert length(agents) == length(x) "Length of effort vector must match number of agents."
    @assert T ≥ 1 "Must have positive number of rounds."
    @assert caching in [:none, :exact, :approximate] "Caching must be :none, :exact, or :approximate."
    num_agents = length(agents)
    # Create matrices
    efforts = zeros(num_agents, T)
    winners = falses(num_agents, T)
    utilities = zeros(num_agents, T)
    nash_gaps = zeros(num_agents, T)
    # Set initial efforts
    efforts[:,1] .= x
    
    # Initialize workspace
    other_efforts_buffer = zeros(num_agents, T)
    total_efforts_buffer = zeros(T)
    
    # Pre-compute constants for workspace
    min_total_efforts = sum(agent.χ for agent in agents)
    max_total_efforts = sum(agent.max_effort for agent in agents)
    min_other_efforts = [min_total_efforts - agent.χ for agent in agents]
    max_other_efforts = [max_total_efforts - agent.max_effort for agent in agents]
    current_round = 0
    
    # Initialize QuadGK buffers for all agents
    # Create specific buffer for Float64, leave Dual buffer as nothing (will be created on first use)
    bayesian_segbufs_float = [QuadGK.alloc_segbuf(Float64, Float64, Float64) for _ in 1:length(agents)]
    bayesian_segbufs_dual = [nothing for _ in 1:length(agents)]  # Will be created on first Dual use
    estimator_segbufs = [QuadGK.alloc_segbuf(Float64, Float64, Float64) for _ in 1:length(agents)]
    
    workspace = ContestWorkspace(
        other_efforts_buffer,
        total_efforts_buffer,
        min_total_efforts,
        max_total_efforts,
        min_other_efforts,
        max_other_efforts,
        caching,
        cache_tolerance,
        bayesian_segbufs_float,
        bayesian_segbufs_dual,
        estimator_segbufs,
        current_round
    )
    
    return TullockContest(agents, efforts, winners, utilities, nash_gaps, workspace)
end


"""  
    nash_gap(contest::TullockContest, t::Int) -> Float64

Compute the total Nash gap for all agents in round `t`.

The Nash gap measures how far the contest is from Nash equilibrium. For each agent,
it's the difference between their maximum possible utility (given others' efforts)
and their actual utility. The total Nash gap is the sum across all agents.

# Arguments
- `contest::TullockContest`: The contest to analyze
- `t::Int`: Round number (1 ≤ t ≤ num_rounds(contest))

# Returns
- `Float64`: Total Nash gap across all agents (≥ 0)

# Convergence
A Nash gap of 0 indicates perfect Nash equilibrium. In practice, the contest
converges when this value drops below a specified tolerance ε.
"""
function nash_gap(contest::TullockContest, t::Int)
    return sum(contest.nash_gaps[:,t])
end