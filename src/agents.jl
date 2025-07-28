"""
An agent has multiple attributes (all immutable)
estimator   — function for estimating opponents' total efforts, is given agent efforts, total effort, and vector of wins
cost        — is a function of agent's effort
p           — probability that agent updates their effort, is a function of the round t
α           — step size, as a function of round t
χ           — lower bound on efforts that agent is permitted to make
max_effort  - upper bound on efforts that agent is permitted to make
h           — the size of the history window, as a function of round t. Return value must be subset of range 1:t-1.
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
Convenience constructor to create an agent with max_effort set automatically.
"""
Agent(est, cost, p, α, χ, h) = Agent(est, cost, p, α, χ, max_agent_effort(cost), h)


"""
Convenience constructor to create an agent with fixed values for
p, α and h in each round. Here h is the size of the history window.
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
Convenience constructor to create an agent with cost function cost(x) = ax^r
with parameters a and r, as well as with fixed values for p, α and h in each round.
"""
function Agent(estimator, p::Float64, α::Float64, χ::Float64, h::Int, a::Float64, r::Float64)
    @assert r ≥ 1 "Exponent of cost function must be ≥ 1."
    cost(x) = a * x^r
    return Agent(estimator, cost, p, α, χ, h)
end


"""
Compute utility of `agent` for their effort `x` given total effort
of other agents `s`.
"""
utility(agent::Agent, x, s::Float64) = x / (x + s) - agent.cost(x)


"""
Compute maximal sensible effort for an agent with `cost` function.
Recall that the reward is normalised to 1, so the agent will never
want to make an effort with a cost greater than 1. Hence, solve
cost(x) = 1 for x using our binary search root finder.
"""
max_agent_effort(cost::Function) = find_root(x->1-cost(x), 0.0)


"""
Return best response of `agent`` to opponents' total effort s.

Implementation makes use of cost functions assumptions that imply
that agent's utility function u(x, s) has a unique maximum on
interval [χ, ∞), so we can find the unique root of its derivative.
"""
function best_response(agent::Agent, s::Float64; _ignore...)
    χ::Float64 = agent.χ
    # Define derivative of utility fn u(z, s) of agent wrt z
    d(z) = ForwardDiff.derivative(z -> utility(agent, z, s), z)
    # If d is negative on entire interval [χ, ∞), return lower boundary χ,
    # otherwise find the root of d, which must lie weakly above χ
    x = d(χ) ≤ 0 ? χ : find_root(d, χ)
    # x = d(χ) ≤ 0 ? χ : find_zero(d, (χ, Inf))
    # x = d(χ) ≤ 0 ? χ : find_zero(d, χ)
    return x
end


"""
Return best response of `agent` to the belief PDF of opponents' total efforts.
"""
function best_response(agent::Agent, pdf::Function;
        min_other_efforts,
        max_other_efforts,
        _ignore...
    )
    χ = agent.χ
    domain = [min_other_efforts, max_other_efforts]
    
    # Define the expected utility function
    function φ(z)
        f = (s, p) -> utility(agent, z, s) * pdf(s)
        prob = IntegralProblem(f, domain)
        return solve(prob, QuadGKJL()).u
    end
    
    # Define derivative of φ
    d(z) = ForwardDiff.derivative(φ, z)
    # If d is negative on entire interval [χ, ∞), return lower boundary χ,
    # otherwise find the root of d, which must lie weakly above χ
    x = d(χ) ≤ 0 ? χ : find_root(d, χ)
    return x
end


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

# Best response caching system (estimator caching removed as ineffective)
const BEST_RESPONSE_CACHE = Dict{Tuple, Float64}()

"""
Clear the best response cache used for optimal effort calculations.
Main benefit: reuse computations when no agents update in a round.
"""
function clear_best_response_cache!()
    empty!(BEST_RESPONSE_CACHE)
    return nothing
end

# Deprecated aliases for backwards compatibility
clear_agent_caches! = clear_best_response_cache!
clear_estimator_cache!() = nothing  # No-op since estimator caching removed


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