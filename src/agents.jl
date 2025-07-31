"""
    Agent

An adaptive agent in a Tullock contest with configurable learning and decision-making behavior.

This immutable struct defines an agent's complete behavioral specification including learning
algorithm, cost structure, participation probability, adaptation rate, effort bounds, and
memory usage. All functions are defined over time to allow for dynamic strategies.

# Fields
- `estimator::Function`: Learning algorithm for estimating opponents' total efforts
  - Signature: `estimator(contest, agent_idx, memory_window, atol, reltol) -> estimate`
  - Given agent's history and outcomes, returns estimated opponent effort
- `cost::Function`: Agent's cost function for effort levels
  - Signature: `cost(effort) -> cost`
  - Should be increasing and convex for theoretical guarantees
- `p::Function`: Probability of updating effort in each round
  - Signature: `p(round) -> probability ∈ [0,1]`
  - Allows for time-varying participation patterns
- `α::Function`: Learning step size/adaptation rate
  - Signature: `α(round) -> step_size ∈ [0,1]`
  - Controls how much agent adapts toward best response (1 = full adaptation)
- `χ::Float64`: Minimum effort bound (must be positive)
  - Prevents degenerate zero-effort solutions
- `max_effort::Float64`: Maximum effort bound
  - Computed automatically from cost function via `max_agent_effort(cost)`
- `h::Function`: Historical memory window function
  - Signature: `h(round) -> range` where range ⊆ 1:round-1
  - Determines which past rounds the agent considers for learning

# Agent Design Principles
- **Modularity**: Each component (learning, costs, adaptation) is independently configurable
- **Time-varying behavior**: All parameters can change over time for complex strategies
- **Bounded rationality**: Finite memory windows and probabilistic updates model realistic limitations
- **Performance optimization**: Functions are called efficiently with minimal allocations

# Theoretical Assumptions
For convergence guarantees, cost functions should satisfy:
- Increasing: c'(x) > 0 for x > 0
- Convex: c''(x) ≥ 0 for x ≥ 0
- Sufficient growth: lim_{x→∞} c(x) = ∞
"""
struct Agent
    estimator::Function
    cost::Function
    p::Function
    α::Function
    χ::Float64
    max_effort::Float64
    h::Function
end


"""
    Agent(estimator, cost, p, α, χ, h) -> Agent

Convenience constructor that automatically computes maximum effort bound.

This constructor calls `max_agent_effort(cost)` to determine the maximum sensible
effort level based on the cost function, eliminating the need to specify it manually.

# Arguments
- `estimator::Function`: Learning algorithm function
- `cost::Function`: Cost function
- `p::Function`: Participation probability function
- `α::Function`: Adaptation rate function
- `χ::Float64`: Minimum effort bound
- `h::Function`: Memory window function

# See Also
- [`max_agent_effort`](@ref): Function that computes maximum effort bound
"""
Agent(est, cost, p, α, χ, h) = Agent(est, cost, p, α, χ, max_agent_effort(cost), h)


"""
    Agent(estimator, cost, p, α, χ, h) -> Agent

Convenience constructor for agents with constant parameters over time.

This constructor creates an agent with fixed participation probability, adaptation rate,
and memory window size that remain constant across all rounds. The memory window is
interpreted as a sliding window of the most recent `h` rounds.

# Arguments
- `estimator::Function`: Learning algorithm function
- `cost::Function`: Cost function c(x) for effort x
- `p::Float64`: Fixed participation probability ∈ [0,1]
- `α::Float64`: Fixed adaptation rate ∈ [0,1]
- `χ::Float64`: Minimum effort bound (> 0)
- `h::Int`: Fixed memory window size (≥ 1)

# Parameter Interpretation
- `p = 0`: Agent never updates (static effort)
- `p = 1`: Agent always considers updating
- `α = 0`: Agent never adapts (ignores best response)
- `α = 1`: Agent fully adapts to best response
- `h = 1`: Agent only remembers the previous round
- `h = ∞`: Agent remembers entire history (use large value)

# Memory Window
The memory window at round t covers rounds max(1, t-h):t-1, ensuring:
- At least 1 round of history when available
- At most h rounds of history
- Never includes the current round t
"""
function Agent(estimator::Function, cost::Function, p::Float64, α::Float64, χ::Float64, h::Int)
    @assert 0 ≤ p ≤ 1 "Probability p must lie between 0 and 1."
    @assert 0 ≤ α ≤ 1 "Convex coefficient α must lie between 0 and 1."
    @assert h ≥ 1 "History window size must be positive."
    p_fn(t::Int)::Float64 = p
    h_fn(t::Int) = max(1, t-h):t-1
    α_fn(t::Int)::Float64 = α
    return Agent(estimator, cost, p_fn, α_fn, χ, h_fn)
end


"""
    Agent(estimator, p, α, χ, h, a, r) -> Agent

Convenience constructor for agents with power-law cost functions.

Creates an agent with cost function c(x) = a⋅x^r and constant behavioral parameters.
Power-law cost functions are common in economic models and provide flexible marginal
cost structures.

# Arguments
- `estimator::Function`: Learning algorithm function  
- `p::Float64`: Participation probability ∈ [0,1]
- `α::Float64`: Adaptation rate ∈ [0,1]
- `χ::Float64`: Minimum effort bound (> 0)
- `h::Int`: Memory window size (≥ 1)
- `a::Float64`: Cost scaling factor (> 0)
- `r::Float64`: Cost exponent (≥ 1)

# Cost Function Properties
- `r = 1`: Linear costs (constant marginal cost)
- `r = 2`: Quadratic costs (linear marginal cost)
- `r > 2`: Superquadratic costs (increasing marginal cost)
- Higher `a`: More expensive effort
- Higher `r`: Steeper cost growth

# Requirements
- `r ≥ 1` ensures convexity for theoretical guarantees
- `a > 0` ensures positive costs for positive effort

# Example
```julia
# Quadratic cost agent: c(x) = 0.5x²
agent = Agent(max_likelihood_estimator, 1.0, 1.0, 0.01, 10, 0.5, 2.0)
```
"""
function Agent(estimator, p::Float64, α::Float64, χ::Float64, h::Int, a::Float64, r::Float64)
    @assert r ≥ 1 "Exponent of cost function must be ≥ 1."
    cost(x) = a * x^r
    return Agent(estimator, cost, p, α, χ, h)
end


"""
    utility(agent::Agent, x, s::Float64) -> Float64

Compute the agent's utility for effort level `x` given opponents' total effort `s`.

The utility function combines the agent's expected payoff from winning (based on
effort shares in the Tullock contest) minus their cost of effort.

# Formula
```
utility(x, s) = x/(x + s) - cost(x)
```

# Arguments
- `agent::Agent`: The agent (for accessing cost function)
- `x`: Agent's effort level (≥ 0)
- `s::Float64`: Total effort of all other agents (≥ 0)

# Returns
- `Float64`: Net utility (can be negative if costs exceed expected payoff)

# Interpretation
- First termn `x/(x + s)`: Probability of winning (effort share)
- Second term `cost(x)`: Cost of exerting effort x
- Utility maximization drives agent to optimal effort given opponents' efforts

# Properties
- Utility is typically concave in own effort x
- At optimum: marginal benefit = marginal cost
- Boundary solutions occur when costs are too high relative to winning probability
"""
utility(agent::Agent, x, s::Float64) = x / (x + s) - agent.cost(x)


"""
    max_agent_effort(cost::Function) -> Float64

Compute the maximum sensible effort level for an agent with given cost function.

Since the contest prize is normalized to 1, no rational agent would exert effort
costing more than the maximum possible payoff. This function finds the effort level
where cost(x) = 1, representing the upper bound on sensible effort.

# Arguments
- `cost::Function`: Agent's cost function c(x)

# Returns
- `Float64`: Maximum effort level where cost equals maximum prize (1.0)

# Implementation
Solves the equation cost(x) = 1 using binary search root finding, which is
equivalent to finding the root of f(x) = 1 - cost(x) starting from x = 0.

# Assumptions
- Cost function is increasing: c'(x) > 0
- Cost function is continuous
- cost(0) < 1 (otherwise no positive effort is viable)
- lim_{x→∞} cost(x) > 1 (solution exists)

# Usage
This function is called automatically by Agent constructors to set appropriate
effort bounds, ensuring agents don't waste computational time considering
uneconomical effort levels.
"""
max_agent_effort(cost::Function) = find_root(x->1-cost(x), 0.0)


"""
    best_response(agent::Agent, s::Float64, _ignore...) -> Float64

Compute the agent's optimal effort level given opponents' total effort.

This function finds the effort level that maximizes the agent's utility given
a fixed level of opponent effort. It uses automatic differentiation to find
where the marginal benefit equals marginal cost.

# Arguments
- `agent::Agent`: The agent making the decision
- `s::Float64`: Total effort of all opponents (≥ 0)
- `_ignore...`: Unused arguments for compatibility with other method signatures

# Returns
- `Float64`: Optimal effort level (≥ agent.χ)

# Algorithm
1. Compute the derivative of utility function with respect to effort
2. If derivative is negative at minimum bound χ, return χ (boundary solution)
3. Otherwise, find the root of the derivative (interior solution)

# Mathematical Foundation
The utility function u(x,s) = x/(x+s) - cost(x) has derivative:
```
u'(x,s) = s/(x+s)² - cost'(x)
```
The optimal effort satisfies u'(x*,s) = 0, meaning marginal benefit = marginal cost.

# Assumptions
- Cost function is convex (ensures unique maximum)
- Utility function has at most one interior maximum
- If no interior solution exists, boundary solution at χ is optimal
"""
function best_response(agent::Agent, s::Float64, _ignore...)
    χ::Float64 = agent.χ
    # Define derivative of utility fn u(z, s) of agent wrt z
    d(z) = ForwardDiff.derivative(z -> utility(agent, z, s), z)
    # If d is negative on entire interval [χ, ∞), return lower boundary χ,
    # otherwise find the root of d, which must lie weakly above χ
    x = d(χ) ≤ 0 ? χ : find_root(d, χ)
    return x
end


"""
    best_response(agent::Agent, pdf::Function, workspace::ContestWorkspace, agent_idx::Int) -> Float64

Return the optimal effort level for `agent` given a belief PDF about opponents' total efforts.

This function computes the expected utility-maximizing effort by integrating over the belief 
distribution using numerical integration. The integration tolerances are controlled by the 
workspace settings for performance optimization.

# Arguments
- `agent::Agent`: The agent making the decision
- `pdf::Function`: Probability density function of opponents' total efforts
- `workspace::ContestWorkspace`: Contest workspace containing integration tolerances and bounds
- `agent_idx::Int`: Index of the agent (used for accessing pre-computed bounds)

# Returns
- `Float64`: Optimal effort level (≥ agent.χ)

# Implementation
Uses QuadGKJL for numerical integration with workspace-configured tolerances (atol, reltol).
The integration domain is pre-computed as [min_other_efforts, max_other_efforts] for efficiency.

# Performance
Integration tolerances can be adjusted via the contest's accuracy parameter:
- `:default`: Highest precision, slowest
- `:relaxed`: Balanced precision/speed  
- `:veryrelaxed`: Lower precision, fastest
"""
function best_response(agent::Agent, pdf::Function, workspace::ContestWorkspace, agent_idx::Int, atol::Float64, rtol::Float64)
    χ = agent.χ
    domain = [workspace.min_other_efforts[agent_idx], workspace.max_other_efforts[agent_idx]]
    
    # Define the expected utility function
    function φ(z)
        f = s -> utility(agent, z, s) * pdf(s)
        # Use appropriate pre-allocated buffer based on type
        if z isa Float64
            segbuf = workspace.bayesian_segbufs_float[agent_idx]
        else
            # Create Dual buffer on first use with correct types
            if workspace.bayesian_segbufs_dual[agent_idx] === nothing
                domain_type = Float64
                range_type = typeof(z)
                error_type = typeof(abs(z))
                workspace.bayesian_segbufs_dual[agent_idx] = QuadGK.alloc_segbuf(domain_type, range_type, error_type)
            end
            segbuf = workspace.bayesian_segbufs_dual[agent_idx]
        end
        
        result, error = quadgk(f, domain[1], domain[2]; 
                              atol=atol, 
                              rtol=rtol,
                              segbuf=segbuf)
        return result
    end
    
    # Define derivative of φ
    d(z) = ForwardDiff.derivative(φ, z)
    # If d is negative on entire interval [χ, ∞), return lower boundary χ,
    # otherwise find the root of d, which must lie weakly above χ
    x = d(χ) ≤ 0 ? χ : find_root_with_caching(d, χ, workspace)
    return x
end


"""
    nash_gap(agent::Agent, effort, other_efforts) -> Float64

Compute the Nash gap for a single agent given current effort levels.

The Nash gap measures how much utility the agent loses by playing their current
effort instead of their best response. It's zero at Nash equilibrium and positive
when the agent could improve by unilaterally changing their effort.

# Arguments
- `agent::Agent`: The agent to analyze
- `effort`: Agent's current effort level
- `other_efforts`: Total effort of all other agents

# Returns
- `Float64`: Nash gap (≥ 0), where 0 indicates optimal play

# Formula
```
Nash gap = utility(best_response, other_efforts) - utility(current_effort, other_efforts)
```

# Interpretation
- Gap = 0: Agent is playing optimally (Nash equilibrium)
- Gap > 0: Agent could improve utility by changing effort
- Larger gaps indicate further deviation from equilibrium
"""
function nash_gap(agent::Agent, effort, other_efforts)
    br = best_response(agent, other_efforts)
    max_utility = utility(agent, br, other_efforts)
    u = utility(agent, effort, other_efforts)
    return max_utility - u
end



# Pre-defined constant functions for better type stability
const ALWAYS_PLAY = (t::Int) -> 1.0::Float64
const FULL_STEP = (t::Int) -> 1.0::Float64
const FULL_HISTORY = (t::Int) -> 1:t-1

# Deprecated aliases for backwards compatibility
clear_agent_caches!() = nothing
clear_estimator_cache!() = nothing


### Constructors for specific kinds of agents


"""
    MLEAgent(cost; χ=0.01) -> Agent

Create a Maximum Likelihood Estimation agent for Tullock contest simulation.

This agent uses maximum likelihood estimation to learn about opponents' total effort 
based on win/loss history. It solves the equation ∑(xᵢ/(xᵢ + y)) = w for the 
estimated opponent effort y, where xᵢ are the agent's efforts and w is the number of wins.

# Arguments
- `cost::Function`: The agent's cost function c(x) where x is effort level
- `χ::Float64=0.01`: Minimum effort bound (must be positive)

# Returns
- `Agent`: Configured agent with MLE learning behavior

# Behavior
- **Participation**: Plays in every round (p(t) = 1)
- **Learning**: Uses entire history for estimation (h(t) = 1:t-1)  
- **Adaptation**: Full commitment to best response (α(t) = 1)
- **Estimation**: Maximum likelihood based on win counts

# Example
```julia
# Linear cost agent
linear_agent = MLEAgent(x -> x, χ=0.02)

# Quadratic cost agent  
quad_agent = MLEAgent(x -> 0.5*x^2)

# Custom cost with higher minimum effort
custom_agent = MLEAgent(x -> x^1.5, χ=0.05)
```

# See Also
- [`DetMLEAgent`](@ref): Deterministic MLE based on observed efforts
- [`BayesianAgent`](@ref): Bayesian learning approach
- [`max_likelihood_estimator`](@ref): The underlying estimation function
"""
function MLEAgent(cost; χ=0.01)
    estimator = max_likelihood_estimator
    return Agent(estimator, cost, ALWAYS_PLAY, FULL_STEP, χ, FULL_HISTORY)
end


"""
    DetMLEAgent(cost; χ=0.01) -> Agent

Create a Deterministic Maximum Likelihood Estimation agent for Tullock contest simulation.

This agent uses deterministic MLE based on observed total efforts rather than just win/loss 
outcomes. It computes expected wins as w = ∑(xᵢ/totalᵢ) and then solves ∑(xᵢ/(xᵢ + y)) = w 
for the estimated opponent effort y.

# Arguments
- `cost::Function`: The agent's cost function c(x) where x is effort level
- `χ::Float64=0.01`: Minimum effort bound (must be positive)

# Returns
- `Agent`: Configured agent with deterministic MLE learning behavior

# Behavior
- **Participation**: Plays in every round (p(t) = 1)
- **Learning**: Uses entire history for estimation (h(t) = 1:t-1)
- **Adaptation**: Full commitment to best response (α(t) = 1)
- **Estimation**: Deterministic MLE based on observed total efforts

# Differences from MLEAgent
- Uses observed total efforts instead of just win counts
- More information-efficient but requires observing all efforts
- Often converges faster than standard MLE

# Example
```julia
cost(x) = x^2
det_agent = DetMLEAgent(cost, χ=0.015)
```

# See Also
- [`MLEAgent`](@ref): Standard MLE based on win counts only
- [`deterministic_max_likelihood_estimator`](@ref): The underlying estimation function
"""
function DetMLEAgent(cost; χ=0.01)
    estimator = deterministic_max_likelihood_estimator
    return Agent(estimator, cost, ALWAYS_PLAY, FULL_STEP, χ, FULL_HISTORY)
end


"""
    DumbAgent(cost; χ=0.01) -> Agent

Create a "dumb" agent that uses simple averaging for Tullock contest simulation.

This agent uses naive estimation by simply averaging observed opponent efforts over 
time, without sophisticated learning algorithms. Despite its simplicity, it often 
performs surprisingly well in practice.

# Arguments
- `cost::Function`: The agent's cost function c(x) where x is effort level
- `χ::Float64=0.01`: Minimum effort bound (must be positive)

# Returns
- `Agent`: Configured agent with simple averaging behavior

# Behavior
- **Participation**: Plays in every round (p(t) = 1)
- **Learning**: Uses entire history for estimation (h(t) = 1:t-1)
- **Adaptation**: Full commitment to best response (α(t) = 1)
- **Estimation**: Simple average of observed opponent efforts

# Characteristics
- Computationally efficient (no complex optimization)
- Robust to noise and outliers
- Good baseline for comparison with sophisticated agents
- Often surprisingly competitive in practice

# Example
```julia
# Simple linear cost dumb agent
dumb_agent = DumbAgent(x -> x)

# Quadratic cost with higher minimum effort
robust_agent = DumbAgent(x -> 0.5*x^2, χ=0.02)
```

# See Also
- [`MLEAgent`](@ref): More sophisticated MLE learning
- [`dumb_estimator`](@ref): The underlying estimation function
"""
function DumbAgent(cost; χ=0.01)
    estimator = dumb_estimator
    return Agent(estimator, cost, ALWAYS_PLAY, FULL_STEP, χ, FULL_HISTORY)
end


"""
    StandardAgent(cost; α=1, χ=0.01) -> Agent

Create a "standard" agent that uses classic exponential smoothing estimation.

This agent uses traditional exponential smoothing to estimate opponent efforts,
providing a middle ground between simple averaging and sophisticated learning algorithms.

# Arguments
- `cost::Function`: The agent's cost function c(x) where x is effort level
- `α::Float64=1`: Step size for learning (1 = full adaptation, 0 = no learning)
- `χ::Float64=0.01`: Minimum effort bound (must be positive)

# Returns
- `Agent`: Configured agent with classic estimation behavior

# Behavior
- **Participation**: Plays in every round (p(t) = 1)
- **Learning**: Uses entire history for estimation (h(t) = 1:t-1)
- **Adaptation**: Configurable step size (α(t) = α)
- **Estimation**: Exponential smoothing of observed efforts

# Step Size Parameter
- `α = 1`: Full commitment to best response (like other agents)
- `α = 0.5`: Moderate adaptation, more robust to noise
- `α = 0.1`: Conservative learning, slow but stable

# Example
```julia
# Full adaptation agent
standard_agent = StandardAgent(x -> x^2)

# Conservative learning agent
conservative_agent = StandardAgent(x -> x, α=0.3, χ=0.02)
```

# See Also
- [`MLEAgent`](@ref): Maximum likelihood estimation approach
- [`classic_estimator`](@ref): The underlying estimation function
"""
function StandardAgent(cost; α=1, χ=0.01)
    estimator = classic_estimator
    α_fn(t::Int)::Float64 = α  # step size function with proper typing
    return Agent(estimator, cost, ALWAYS_PLAY, α_fn, χ, FULL_HISTORY)
end



"""
    BayesianAgent(cost; χ=0.01) -> Agent

Create a Bayesian learning agent for Tullock contest simulation.

This agent uses sophisticated Bayesian learning with numerical integration to maintain 
probability distributions over opponent behavior. It provides theoretically optimal 
learning but is computationally expensive.

# Arguments
- `cost::Function`: The agent's cost function c(x) where x is effort level
- `χ::Float64=0.01`: Minimum effort bound (must be positive)

# Returns
- `Agent`: Configured agent with Bayesian learning behavior

# Behavior
- **Participation**: Plays in every round (p(t) = 1)
- **Learning**: Uses entire history for estimation (h(t) = 1:t-1)
- **Adaptation**: Full commitment to best response (α(t) = 1)
- **Estimation**: Bayesian updating with numerical integration

# Characteristics
- Theoretically optimal learning approach
- Maintains full probability distributions over beliefs
- Computationally expensive (10-100x slower than other agents)
- Most sophisticated learning algorithm available
- Accounts for uncertainty in opponent behavior

# Performance Notes
- Significantly slower than other agent types
- Requires numerical integration for each decision
- Best suited for small contests or research applications
- Consider other agents for large-scale simulations

# Example
```julia
# Basic Bayesian agent
bayesian_agent = BayesianAgent(x -> x^2)

# High-precision Bayesian agent
precise_agent = BayesianAgent(x -> 0.5*x^1.5, χ=0.005)
```

# See Also
- [`MLEAgent`](@ref): Faster maximum likelihood approach
- [`bayesian_estimator`](@ref): The underlying estimation function
- [`best_response`](@ref): Optimization with probability distributions
"""
function BayesianAgent(cost; χ=0.01)
    estimator = bayesian_estimator
    return Agent(estimator, cost, ALWAYS_PLAY, FULL_STEP, χ, FULL_HISTORY)
end