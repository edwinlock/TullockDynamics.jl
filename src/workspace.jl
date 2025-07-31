"""
    ContestWorkspace

Workspace for intermediate computations and configuration in Tullock contests.

This mutable struct contains pre-allocated buffers, pre-computed constants, and configuration
settings to optimize contest simulation performance. It eliminates repeated memory allocations
and expensive computations during simulation.

# Fields

## Pre-allocated Buffers (updated each round)
- `other_efforts::Matrix{Float64}`: Buffer storing total other efforts for each agent per round
- `total_efforts::Vector{Float64}`: Total effort across all agents in each round

## Pre-computed Constants (computed once at initialization)
- `min_total_efforts::Float64`: Sum of all agents' minimum effort bounds (χ values)
- `max_total_efforts::Float64`: Sum of all agents' maximum effort bounds
- `min_other_efforts::Vector{Float64}`: For each agent, sum of other agents' minimum efforts
- `max_other_efforts::Vector{Float64}`: For each agent, sum of other agents' maximum efforts

## Root Finding Configuration
- `caching::Symbol`: Root finding caching strategy (:none, :exact, or :approximate)
- `cache_tolerance::Float64`: Tolerance for approximate caching

## Simulation State
- `current_round::Int`: Current round number (for tracking simulation progress)

# Performance Features
- **Memory efficiency**: Pre-allocated buffers eliminate allocations during simulation
- **Computational efficiency**: Pre-computed bounds avoid repeated calculations  
- **Configurable precision**: Adjustable integration tolerances for performance/accuracy tradeoff
- **Cache-friendly**: Optimized data layout for better memory access patterns

# Integration Tolerances
Integration tolerances are now passed dynamically to the `run!` function via the `accuracy` parameter.
This allows the same contest to be run with different precision levels without reconstruction.
"""
mutable struct ContestWorkspace
    # Pre-allocated buffers updated in every round
    other_efforts::Matrix{Float64}          # Buffer for the total other efforts, for each agent i in each round
    total_efforts::Vector{Float64}          # Total effort in each round
    
    # Pre-computed constants (computed once in constructor)
    min_total_efforts::Float64              # sum of agents' minimal efforts
    max_total_efforts::Float64              # sum of agents' maximal efforts 
    min_other_efforts::Vector{Float64}      # for each agent, sum of other agents' minimal efforts
    max_other_efforts::Vector{Float64}      # for each agent, sum of other agents' maximal efforts

    # Root finding configuration  
    caching::Symbol                         # Root finding caching strategy: :none, :exact, or :approximate (default: :approximate)
    cache_tolerance::Float64                # Tolerance for approximate caching (default: 1e-9)
    
    # QuadGK buffer management (two specific buffers per agent)
    bayesian_segbufs_float::Vector{Any}   # Float64 buffers for best_response
    bayesian_segbufs_dual::Vector{Any}    # ForwardDiff.Dual buffers for best_response  
    estimator_segbufs::Vector{Any}        # Float64 buffers for estimator (only needs Float64)
    
    # Current round
    current_round::Int
end