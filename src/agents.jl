"""
An agent has multiple attributes (all immutable)
cost        — is a function of agent's effort
p           — probability that agent updates their effort, is a function of the round t
h           — the size of the history window, as a function of round t. Return value must be subset of range 1:t-1.
α           — step size, as a function of round t
estimator   — function for estimating opponents' total efforts, is given agent efforts, total effort, and vector of wins
χ           — lower bound on efforts that agent is permitted to make
max_effort  - upper bound on efforts that agent is permitted to make
"""
struct Agent
    estimator::Function
    cost::Function
    p::Function
    α::Function
    h::Function
    χ::Float64
    max_effort::Float64
end


"""
Convenience constructor to create an agent with fixed values for
p, α and h in each round. Here h is the size of the history window.
"""
function Agent(estimator::Function, cost::Function, p::Float64, α::Float64, h::Int, χ::Float64)
    @assert 0 ≤ p ≤ 1 "Probability p must lie between 0 and 1."
    @assert 0 ≤ α ≤ 1 "Convex coefficient α must lie between 0 and 1."
    @assert h ≥ 1 "History window size must be positive."
    p_fn(t) = p
    h_fn(t) = max(1, t-h):t-1
    α_fn(t) = α
    max_effort = max_effort(cost)
    return Agent(estimator, cost, p_fn, α_fn, h_fn, χ, max_effort)
end


"""
Convenience constructor to create an agent with cost function cost(x) = ax^r
with parameters a and r, as well as with fixed values for p, α and h in each round.
"""
function Agent(estimator, p::Float64, α::Float64, h::Int, χ::Float64, a::Float64, r::Float64)
    @assert r ≥ 1 "Exponent of cost function must be ≥ 1."
    cost(x) = a * x^r
    return Agent(estimator, cost, p_fn, α_fn, h_fn, χ)
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
max_effort(cost) = find_root(x->1-cost(x), 0.0)


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
    # Define the function to optimise over:
    domain = [min_other_efforts, max_other_efforts]
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



### Constructors for specific kinds of agents


"""
Create an MLE agent with `cost` function, and minimum effort
bound `χ` (with default 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function MLEAgent(cost; χ=0.01)
    estimator = max_likelihood_estimator
    p(t) = 1  # plays in every round
    α(t) = 1  # commits fully to best response (no interpolation)
    h(t) = 1:t-1  # access to entire history
    return Agent(estimator, cost, p, α, h, χ)
end


"""
Create a deterministic MLE agent with `cost` function, and minimum
effort bound `χ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function DetMLEAgent(cost; χ=0.01)
    estimator = deterministic_max_likelihood_estimator
    p(t) = 1  # plays in every round
    α(t) = 1  # commits fully to best response (no interpolation)
    h(t) = 1:t-1  # access to entire history
    return Agent(estimator, cost, p, α, h, χ)
end


"""
Create a dumb agent with `cost` function, and minimum
effort bound `χ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function DumbAgent(cost; χ=0.01)
    estimator = dumb_estimator
    p(t) = 1  # plays in every round
    α(t) = 1  # commits fully to best response (no interpolation)
    h(t) = 1:t-1  # access to entire history
    return Agent(estimator, cost, p, α, h, χ)
end


"""
Create a classic agent with `cost` function, and minimum
effort bound `χ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size `α` (defaults to 1).
"""
function StandardAgent(cost; α=1, χ=0.01)
    estimator = omniscient_estimator
    p(t) = 1  # plays in every round
    step(t) = 1  # commits fully to best response (no interpolation)
    h(t) = 1:t-1  # access to entire history
    return Agent(estimator, cost, p, step, h, χ)
end



"""
Create a Bayesian agent with `cost` function, and minimum effort
bound `χ` (with default 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function BayesianAgent(cost; χ=0.01)
    estimator = bayesian_estimator
    p(t) = 1  # plays in every round
    α(t) = 1  # commits fully to best response (no interpolation)
    h(t) = 1:t-1  # access to entire history
    return Agent(estimator, cost, p, α, h, χ, max_effort(cost))
end