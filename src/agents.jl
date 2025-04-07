"""
An agent has multiple attributes (all immutable)
cost        — is a function of agent's effort
p           — probability that agent updates their effort, is a function of the round t
window      — the size of the memory window, as a function of round t. Return value must be subset of range 1:t-1.
α           — step size, as a function of round t
estimator   — function for estimating opponents' total efforts, is given agent efforts, total effort, and vector of wins
δ           — lower bound on efforts that agent is permitted to make
"""
struct Agent
    cost::Function
    p::Function
    window::Function
    α::Function
    estimator::Function
    δ::Float64
end


"""
Compute utility of `agent` for their effort `x` given total effort
of other agents `s`.
"""
utility(agent::Agent, x, s::Float64) = x / (x + s) - agent.cost(x)


"""
Return best response of `agent`` to opponents' total effort s.

Implementation makes use of cost functions assumptions that imply
that agent's utility function u(x, s) has a unique maximum on
interval [δ, ∞), so we can find the unique root of its derivative.
"""
function best_response(agent::Agent, s::Float64)
    δ::Float64 = agent.δ
    # Define derivative of utility fn u(z, s) of agent wrt z
    d(z) = ForwardDiff.derivative(z -> utility(agent, z, s), z)
    # If d is negative on entire interval [δ, ∞), return lower boundary δ,
    # otherwise find the root of d, which must lie weakly above δ
    x = d(δ) ≤ 0 ? δ : find_root(d, δ)
    # x = d(δ) ≤ 0 ? δ : find_zero(d, (δ, Inf))
    # x = d(δ) ≤ 0 ? δ : find_zero(d, δ)
    return x
end


"""
Return best response of `agent` to the belief PDF of opponents' total efforts.
"""
function best_response(agent::Agent, pdf::Function)
    δ = agent.δ
    # Define the function to optimise over:
    domain = [0, Inf]
    function φ(z)
        f = (s, p) -> utility(agent, z, s) * pdf(s)
        prob = IntegralProblem(f, domain)
        return solve(prob, QuadGKJL()).u
    end
    # Define derivative of φ
    d(z) = ForwardDiff.derivative(φ, z)
    # If d is negative on entire interval [δ, ∞), return lower boundary δ,
    # otherwise find the root of d, which must lie weakly above δ
    x = d(δ) ≤ 0 ? δ : find_root(d, δ)
    # x = d(δ) ≤ 0 ? δ : find_zero(d, (δ, Inf))
    # x = d(δ) ≤ 0 ? δ : find_zero(d, δ)
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
bound `δ` (with default 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function MLEAgent(cost; δ=0.01)
    p(t) = 1  # plays in every round
    window(t) = 1:t-1  # access to entire history
    α(t) = 1  # commits fully to best response (no interpolation)
    estimator = max_likelihood_estimator
    return Agent(cost, p, window, α, estimator, δ)
end


"""
Create a deterministic MLE agent with `cost` function, and minimum
effort bound `δ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function DetMLEAgent(cost; δ=0.01)
    p(t) = 1  # plays in every round
    window(t) = 1:t-1  # access to entire history
    α(t) = 1  # commits fully to best response (no interpolation)
    estimator = deterministic_max_likelihood_estimator
    return Agent(cost, p, window, α, estimator, δ)
end


"""
Create a dumb agent with `cost` function, and minimum
effort bound `δ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size 1.
"""
function DumbAgent(cost; δ=0.01)
    p(t) = 1  # plays in every round
    window(t) = 1:t-1  # access to entire history
    α(t) = 1  # commits fully to best response (no interpolation)
    estimator = dumb_estimator
    return Agent(cost, p, window, α, estimator, δ)
end


"""
Create a classic agent with `cost` function, and minimum
effort bound `δ` (defaults to 0.01).

The agent plays in every round, has access to her entire history
and has step size `α` (defaults to 1).
"""
function ClassicAgent(cost; α=1, δ=0.01)
    p(t) = 1  # plays in every round
    window(t) = 1:t-1  # access to entire history
    step(t) = α  # commits fully to best response (no interpolation)
    estimator = omniscient_estimator
    return Agent(cost, p, window, step, estimator, δ)
end
