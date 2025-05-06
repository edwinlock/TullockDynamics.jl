"""
Implementation of maximum likelihood estimator, given own `efforts` and `wins`.
"""
function max_likelihood_estimator(;
        own_efforts::Vector{Float64},
        wins::Vector{Bool},
        cost::Function,
        _ignore...)::Float64
    w = sum(wins)  # compute number of wins
    # Edge case of zero wins: return highest x for which cost doesn't exceed payoff 1.
    w == 0 && return max_agent_effort(cost)
    # Solve sum(x[i] / (x[i] + y)) = w for y:
    # Define strictly decreasing function for root finding
    f(y) = sum(x / (x + y) for x in own_efforts) - w
    return find_root(f, 0.0)
end


"""
Implementation of maximum likelihood estimator, given `own_efforts` and `total_efforts`, in each round.
"""
function deterministic_max_likelihood_estimator(;
        own_efforts::Vector{Float64},
        total_efforts::Vector{Float64},
        _ignore...)::Float64
    @assert eachindex(own_efforts) == eachindex(total_efforts)
    # Compute expected number of wins
    w = sum(own_efforts[i] / total_efforts[i] for i ∈ eachindex(own_efforts))
    # Solve sum(x[i] / (x[i] + y)) = w for y:
    # Define strictly decreasing function for root finding
    f(y) = sum(x / (x + y) for x in own_efforts) - w
    return find_root(f, 0.0)
end


"""
Implementation of "dumb" estimate, given own `efforts' and `wins`.
"""
function dumb_estimator(;
        own_efforts::Vector{Float64},
        wins::Vector{Bool},
        cost::Function,
        _ignore...)::Float64
    w = sum(wins)  # compute number of wins
    # Edge case of zero wins: return highest x for which cost doesn't exceed payoff 1.
    w == 0 && return max_agent_effort(cost)
    num_rounds = length(own_efforts)
    avg_effort = sum(x for x ∈ own_efforts) / num_rounds
    # Throw error in case of zero effort
    @assert avg_effort > 0 "Average effort cannot be 0!"
    win_fraction = w / num_rounds
    y = avg_effort / win_fraction - avg_effort
    return y
end


"""
Implementation of full knowledge estimate, given own `efforts` and `total_efforts`,
which simply returns the total effort of others in the last round.
"""
function classic_estimator(;
        own_efforts::Vector{Float64},
        total_efforts::Vector{Float64},
        _ignore...)::Float64
    return total_efforts[end] - own_efforts[end]
end


"""
Implementation of Bayesian estimate, given own `efforts` and `wins`.
"""
function bayesian_estimator(;
        own_efforts::Vector{Float64},
        wins::Vector{Bool},
        min_other_efforts::Float64,
        max_other_efforts::Float64,
        _ignore...
    )
    l = count(==(0), wins)  # count number of losses
    lb, ub = min_other_efforts, max_other_efforts
    # Hard-code the uniform distribution on domain (lb, ub)
    initial_pdf(y) = lb ≤ y ≤ ub ? 1. / (ub - lb) : 0.
    # Determine the unnormalised estimator f
    f(y,p) = exp( l*log(y)  + log(initial_pdf(y)) - sum(log(x + y) for x ∈ own_efforts) )
    # Compute the integral of f on domain [lb, ub] to normalise f
    domain = (lb, ub)
    prob = IntegralProblem(f, domain)
    M = solve(prob, QuadGKJL()).u
    # Define the normalised estimator μ
    μ(y) = lb ≤ y ≤ ub ? f(y,0) / M : 0.
    return μ
end
