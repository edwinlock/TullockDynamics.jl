"""
    TullockDynamics

A high-performance Julia package for simulating Tullock contest dynamics with heterogeneous learning agents.

TullockDynamics.jl provides a comprehensive framework for modeling contests where agents compete 
by exerting costly effort, with winning probabilities proportional to relative effort. The package 
features multiple agent types with different learning algorithms, optimized memory management, 
and extensive performance enhancements.

# Key Features

- **Multiple Agent Types**: MLE, Deterministic MLE, Bayesian, and Classic agents
- **High Performance**: Workspace-based memory management and enhanced numerical algorithms  
- **Comprehensive Testing**: 1,300+ tests ensuring correctness and performance
- **Flexible Configuration**: Customizable cost functions, learning rates, and memory windows
- **Analysis Tools**: Built-in visualization and convergence analysis

# Core Components

## Agent Types
- [`MLEAgent`](@ref): Maximum likelihood estimation learning
- [`DetMLEAgent`](@ref): Deterministic MLE based on observed efforts  
- [`BayesianAgent`](@ref): Bayesian learning with numerical integration
- [`DumbAgent`](@ref): Simple effort averaging
- [`Agent`](@ref): Custom agent constructor

## Contest Simulation  
- [`TullockContest`](@ref): Main contest data structure
- [`run!`](@ref): Execute contest dynamics
- [`step!`](@ref): Single round simulation
- [`convergence_status`](@ref): Check convergence state

## Analysis and Utilities
- [`visualise`](@ref): Plot contest trajectories
- [`nash_gap`](@ref): Measure distance from Nash equilibrium
- [`find_root`](@ref): Enhanced root finding for numerical solutions
- [`utility`](@ref): Agent utility calculations

# Quick Start Example

```julia
using TullockDynamics

# Define cost function and create agents
cost(x) = 0.5 * x^2
agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]

# Set up and run contest
contest = TullockContest(agents, [0.1, 0.15, 0.2], 50)
final_round = run!(contest)

# Analyze results
converged, actual_final = convergence_status(contest, final_round)
visualise(contest)
```

# Performance Notes

The package is optimized for high-performance simulations with:
- Workspace-based memory allocation patterns
- Enhanced root finding with multiple termination conditions
- Direct indexing to avoid closure allocations
- Cached computations for frequently used values

Typical performance: Small contests (2-5 agents, 50 rounds) complete in < 10ms.

# Mathematical Background

In Tullock contests, agent i maximizes expected utility:
U_i(x_i, x_{-i}) = x_i / (∑x_j) - c_i(x_i)

The package simulates learning dynamics as agents adapt strategies based on observed outcomes
using various estimation algorithms.
"""
module TullockDynamics

using StatsBase  # used to determine winner in each round
using ForwardDiff  # used to differentiate cost function
using Integrals  # used to compute Bayesian estimates
using Plots
using ProgressMeter
using Measures

include("workspace.jl")
include("agents.jl")
include("contests.jl")
include("dynamics.jl")
include("estimators.jl")
include("utils.jl")

# Exports from agents.jl
export Agent, MLEAgent, DetMLEAgent, DumbAgent, BayesianAgent, StandardAgent
export utility, best_response, nash_gap, max_agent_effort
export clear_best_response_cache!, clear_agent_caches!
# Exports from contests.jl
export TullockContest
export nash_gap, num_rounds
# Exports from dynamics.jl
export set_efforts!, set_utilities!, step!, run!
# Exports from estimators.jl
export max_likelihood_estimator, deterministic_max_likelihood_estimator, dumb_estimator, bayesian_estimator, classic_estimator
export clear_bayesian_cache!
# Exports from utils.jl
export find_root, find_root_cached, find_root_with_caching, show, final_efforts, visualise, convergence_status

end # module TullockDynamics
