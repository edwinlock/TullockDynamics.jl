# TullockDynamics.jl

[![Build Status](https://github.com/edwinlock/TullockDynamics.jl/workflows/CI/badge.svg)](https://github.com/edwinlock/TullockDynamics.jl/actions)
[![Coverage](https://codecov.io/gh/edwinlock/TullockDynamics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/edwinlock/TullockDynamics.jl)

A high-performance Julia package for simulating Tullock contest dynamics with heterogeneous learning agents.

## Overview

TullockDynamics.jl provides a comprehensive framework for modeling and simulating contests where agents compete by exerting costly effort, with the probability of winning proportional to their relative effort (Tullock contest). The package features multiple agent types with different learning algorithms, efficient memory management, and extensive performance optimizations.

## Key Features

- **Multiple Agent Types**: MLE, Deterministic MLE, Bayesian, and Classic agents with different learning strategies
- **High Performance**: Optimized workspace-based memory management and enhanced numerical algorithms
- **Comprehensive Testing**: 1,300+ tests ensuring mathematical correctness and performance
- **Flexible Configuration**: Customizable cost functions, learning rates, and memory windows
- **Rich Analysis Tools**: Built-in visualization and convergence analysis capabilities

## Installation

```julia
using Pkg
Pkg.add("TullockDynamics")
```

Or from the Julia REPL:
```julia
] add TullockDynamics
```

## Quick Start

```julia
using TullockDynamics

# Define a cost function
cost(x) = 0.5 * x^2

# Create agents with different learning strategies
agents = [
    MLEAgent(cost),           # Maximum likelihood estimation
    DetMLEAgent(cost),        # Deterministic MLE  
    DumbAgent(cost),          # Simple averaging
    BayesianAgent(cost)       # Bayesian learning
]

# Set up a contest
initial_efforts = [0.1, 0.15, 0.2, 0.12]
T = 50  # Number of rounds
contest = TullockContest(agents, initial_efforts, T)

# Run the simulation
final_round = run!(contest)

# Check if converged
converged, actual_final = convergence_status(contest, final_round)
println("Converged: $converged after $actual_final rounds")

# Get final efforts and utilities
final_efforts = contest.efforts[:, actual_final]
final_utilities = contest.utilities[:, actual_final]

# Visualize the dynamics
visualise(contest)
```

## Agent Types

### MLEAgent
Uses maximum likelihood estimation to learn about opponents' total effort based on win/loss history.

```julia
agent = MLEAgent(cost; χ=0.01)  # χ is minimum effort bound
```

### DetMLEAgent  
Uses deterministic maximum likelihood estimation based on observed total efforts rather than just wins/losses.

```julia
agent = DetMLEAgent(cost; χ=0.01)
```

### BayesianAgent
Employs Bayesian learning with numerical integration to maintain beliefs about opponent behavior.

```julia
agent = BayesianAgent(cost; χ=0.01)
```

### DumbAgent
Uses simple averaging of observed efforts without sophisticated learning.

```julia
agent = DumbAgent(cost; χ=0.01)
```

### Custom Agents
Create agents with custom learning algorithms:

```julia
agent = Agent(
    estimator_function,    # Function to estimate opponent efforts
    cost_function,         # Agent's cost function  
    play_probability,      # Function determining when to play
    step_size,            # Learning rate function
    minimum_effort,       # Lower bound on effort
    max_effort,           # Upper bound on effort  
    memory_window         # Function defining memory window
)
```

## Advanced Usage

### Custom Cost Functions
```julia
# Linear cost
linear_cost(x) = x

# Quadratic cost  
quadratic_cost(x) = 0.5 * x^2

# Power cost
power_cost(α) = x -> x^α

# Exponential cost
exp_cost(x) = exp(x) - 1
```

### Performance Optimization
The package includes several performance optimizations:

- **Workspace-based memory management** eliminates redundant allocations
- **Enhanced root finding** with multiple termination conditions  
- **Direct indexing** avoids closure allocations in estimators
- **Cached computations** for frequently used values

### Convergence Analysis
```julia
# Check Nash equilibrium properties
nash_gaps = contest.nash_gaps[:, actual_final]
total_gap = sum(nash_gaps)

# Analyze convergence over time
gap_history = [sum(contest.nash_gaps[:, t]) for t in 1:actual_final]
```

## Mathematical Background

In a Tullock contest, agent $i$ chooses effort $x_i$ to maximize expected utility:

$$U_i(x_i, x_{-i}) = \frac{x_i}{\sum_{j=1}^n x_j} - c_i(x_i)$$

where $c_i(x_i)$ is agent $i$'s cost function. The package simulates the learning dynamics as agents adapt their strategies over time based on observed outcomes.

### Learning Algorithms

- **MLE**: Solves $\sum_{t} \frac{x_{i,t}}{x_{i,t} + y} = w$ for estimated opponent effort $y$
- **Deterministic MLE**: Uses observed total efforts rather than win counts
- **Bayesian**: Maintains posterior beliefs over opponent effort distributions  
- **Classic**: Simple exponential smoothing of observed efforts

## Performance

The package is optimized for high-performance simulations:

- **Enhanced root finding**: Custom algorithm with multiple termination conditions
- **Memory efficiency**: Workspace-based allocation patterns minimize garbage collection
- **Numerical stability**: Robust handling of edge cases and extreme parameter values

Typical performance on modern hardware:
- Small contests (2-5 agents, 50 rounds): < 10ms
- Medium contests (10 agents, 100 rounds): < 100ms  
- Large contests (20+ agents): Scales efficiently with optimized algorithms

## Testing

The package includes comprehensive tests covering:

- Mathematical properties and Nash equilibrium conditions
- Edge cases and error handling
- Performance regression detection
- Cross-platform compatibility
- Integration testing with realistic scenarios

Run tests with:
```julia
using Pkg
Pkg.test("TullockDynamics")
```

## Contributing

Contributions are welcome! Please see the issues list for areas needing improvement:

1. **ForwardDiff optimization**: Replace automatic differentiation with analytical derivatives
2. **Bayesian integration caching**: Implement memoization for expensive integrals  
3. **Additional agent types**: New learning algorithms and behavioral models
4. **Visualization enhancements**: Interactive plotting and analysis tools

## Benchmarking

A comprehensive benchmark suite is available in `benchmark_find_root.jl` comparing different root-finding implementations. Run with:

```julia
include("benchmark_find_root.jl")
```

## Citation

If you use TullockDynamics.jl in your research, please cite:

```bibtex
@software{TullockDynamics.jl,
  title = {TullockDynamics.jl: High-Performance Tullock Contest Simulations},
  url = {https://github.com/edwinlock/TullockDynamics.jl},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Related Work

- Tullock, G. (1980). Efficient Rent Seeking. In J. M. Buchanan, R. D. Tollison, & G. Tullock (Eds.), Toward a Theory of the Rent-Seeking Society (pp. 97-112). Texas A&M University Press.
- Konrad, K. A. (2009). Strategy and Dynamics in Contests. Oxford University Press.

## Acknowledgments

This package was developed with significant contributions from Claude Code for performance optimization, testing infrastructure, and mathematical verification.