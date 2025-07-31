
"""
    max_likelihood_estimator(contest::TullockContest, agent_idx::Int, window; _ignore...) -> Float64

Estimate total effort of other agents using maximum likelihood estimation based on win history.

# Arguments
- `contest::TullockContest`: The contest containing all agent data
- `agent_idx::Int`: Index of the agent making the estimate
- `window`: Time window (range or collection) of rounds to use for estimation

# Returns
- `Float64`: Estimated total effort of all other agents

# Algorithm
Solves the equation `∑(xᵢ/(xᵢ + y)) = w` for `y`, where:
- `xᵢ` are the agent's own efforts in the window
- `w` is the number of wins in the window
- `y` is the estimated total effort of others

# Edge Cases
- If the agent has zero wins, returns the maximum feasible effort based on cost function
- Uses binary search root finding for numerical stability
"""
function max_likelihood_estimator(
        contest::TullockContest,
        agent_idx::Int,
        window
    )::Float64

    # Compute number of wins
    w = count(t -> contest.winners[agent_idx, t], window)

    # Edge case of zero wins: return highest x for which cost doesn't exceed payoff 1.
    w == 0 && return max_agent_effort(contest.agents[agent_idx].cost)
    # Solve sum(x[i] / (x[i] + y)) = w for y:
    # Define strictly decreasing function for root finding
    effort_values = @view contest.efforts[agent_idx, window]  # Direct indexing
    
    # Check for edge case: if all effort values are zero
    if all(effort_values .== 0.0)
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    # Check for mathematical validity: when some efforts are zero and w > 0,
    # the function f(y) = sum(x_i/(x_i + y)) - w may be always negative
    max_possible_sum = sum(x > 0 ? 1.0 : 0.0 for x in effort_values)
    if w > max_possible_sum
        # More wins than mathematically possible given the effort pattern
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    f(y) = sum(x / (x + y) for x in effort_values) - w
    
    # Verify f(small_positive) >= 0 before calling find_root
    if f(1e-10) < 0  # Use small positive value to avoid NaN at f(0)
        # Function is always negative, return fallback
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    return find_root_with_caching(f, 0.0, contest.workspace)
end


"""
    deterministic_max_likelihood_estimator(contest::TullockContest, agent_idx::Int, window; _ignore...) -> Float64

Estimate total effort of other agents using deterministic maximum likelihood based on observed total efforts.

# Arguments
- `contest::TullockContest`: The contest containing all agent data
- `agent_idx::Int`: Index of the agent making the estimate
- `window`: Time window (range or collection) of rounds to use for estimation

# Returns
- `Float64`: Estimated total effort of all other agents

# Algorithm
Computes expected number of wins as `w = ∑(xᵢ/totalᵢ)` over the window, then
solves `∑(xᵢ/(xᵢ + y)) = w` for `y`, where:
- `xᵢ` are the agent's own efforts
- `totalᵢ` are the observed total efforts in each round
- `y` is the estimated total effort of others

# Notes
This variant uses deterministic win probabilities rather than actual win outcomes.
"""
function deterministic_max_likelihood_estimator(
        contest::TullockContest,
        agent_idx::Int,
        window
    )::Float64

    ws = contest.workspace

    # Compute expected number of wins directly
    w = 0.0
    for t in window
        own_effort = contest.efforts[agent_idx, t]
        total_effort = ws.total_efforts[t]
        if total_effort > 0.0
            w += own_effort / total_effort
        else
            # If total effort is zero, assume uniform probability
            w += 1.0 / length(contest.agents)
        end
    end
    # Solve sum(x[i] / (x[i] + y)) = w for y:
    # Define strictly decreasing function for root finding
    effort_values = @view contest.efforts[agent_idx, window]
    
    # Check for edge case: if all effort values are zero
    if all(effort_values .== 0.0)
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    # Check for mathematical validity: when some efforts are zero and w > 0,
    # the function f(y) = sum(x_i/(x_i + y)) - w may be always negative
    max_possible_sum = sum(x > 0 ? 1.0 : 0.0 for x in effort_values)
    if w > max_possible_sum
        # More wins than mathematically possible given the effort pattern
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    f(y) = sum(x / (x + y) for x in effort_values) - w

    # Verify f(small_positive) >= 0 before calling find_root
    if f(1e-10) < 0  # Use small positive value to avoid NaN at f(0)
        # Function is always negative, return fallback
        return max_agent_effort(contest.agents[agent_idx].cost)
    end
    
    return find_root_with_caching(f, 0.0, contest.workspace)
end


"""
    dumb_estimator(contest::TullockContest, agent_idx::Int, window; _ignore...) -> Float64

Simple heuristic estimator based on average effort and win rate.

# Arguments
- `contest::TullockContest`: The contest containing all agent data
- `agent_idx::Int`: Index of the agent making the estimate
- `window`: Time window (range or collection) of rounds to use for estimation

# Returns
- `Float64`: Estimated total effort of all other agents

# Algorithm
Uses the heuristic formula: `y = (num_rounds/wins - 1) × avg_effort`, where:
- `num_rounds` is the length of the estimation window
- `wins` is the total number of wins in the window
- `avg_effort` is the agent's average effort over the window

# Edge Cases
- If the agent has zero wins, returns the maximum feasible effort based on cost function
- Requires non-zero average effort (will assert if violated)

# Notes
This is a simple, fast heuristic that doesn't use sophisticated statistical methods.
"""
function dumb_estimator(
        contest::TullockContest,
        agent_idx::Int,
        window
    )::Float64
    
    # Compute number of wins
    w = 0
    for t ∈ window
        w += contest.winners[agent_idx, t]
    end

    # Edge case of zero wins: return highest x for which cost doesn't exceed payoff 1.
    w == 0 && return max_agent_effort(contest.agents[agent_idx].cost)

    # Compute average effort
    sumefforts = 0
    for t ∈ window
        sumefforts += contest.efforts[agent_idx, t]
    end
    num_rounds = length(window)
    avg_effort = sumefforts / num_rounds
    
    # Throw error in case of zero effort
    @assert avg_effort > 0 "Average effort cannot be 0!"

    y = (num_rounds / w - 1.0) * avg_effort
    return y
end


"""
    classic_estimator(contest::TullockContest, agent_idx::Int, window; _ignore...) -> Float64

Perfect information estimator that returns the actual total effort of others in the last round.

# Arguments
- `contest::TullockContest`: The contest containing all agent data
- `agent_idx::Int`: Index of the agent making the estimate
- `window`: Time window (range or collection) of rounds to use for estimation

# Returns
- `Float64`: Actual total effort of all other agents in the last round of the window

# Algorithm
Simply computes `total_effort_last - own_effort_last` from the most recent round.

# Notes
This estimator assumes perfect information about other agents' efforts, making it
useful as a benchmark or for omniscient agents. In practice, agents would not
have access to this information.
"""
function classic_estimator(
        contest::TullockContest,
        agent_idx::Int,
        window
    )::Float64

    ws = contest.workspace

    last_round = window[end]    
    return ws.total_efforts[last_round] - contest.efforts[agent_idx, last_round]
end


"""
    bayesian_estimator(contest::TullockContest, agent_idx::Int, window; _ignore...) -> Function

Bayesian estimator that returns a probability density function over possible opponent efforts.

# Arguments
- `contest::TullockContest`: The contest containing all agent data
- `agent_idx::Int`: Index of the agent making the estimate
- `window`: Time window (range or collection) of rounds to use for estimation

# Returns
- `Function`: A normalized probability density function `μ(y)` over opponent effort `y`

# Algorithm
1. Uses a uniform prior distribution over the feasible effort range
2. Updates belief based on observed losses using Bayesian inference
3. The likelihood function accounts for the probability of losing given opponent effort
4. Returns the normalized posterior distribution

# Mathematical Details
The unnormalized posterior is proportional to:
```
f(y) ∝ y^l × prior(y) × ∏ᵢ 1/(xᵢ + y)
```
where:
- `l` is the number of losses in the window
- `xᵢ` are the agent's own efforts
- The product reflects the likelihood of the observed outcomes

# Performance Notes
Uses numerical integration (QuadGKJL) to normalize the distribution. Integration tolerances
are controlled by the contest's workspace settings, which can be adjusted via the `accuracy`
parameter in `TullockContest` constructor for performance optimization.
"""
function bayesian_estimator(
        contest::TullockContest,
        agent_idx::Int,
        window,
        atol,
        rtol,
    )
    # Compute number of losses
    l = 0
    for t ∈ window
        l += 1 - contest.winners[agent_idx, t]
    end

    lb, ub = contest.workspace.min_other_efforts[agent_idx], contest.workspace.max_other_efforts[agent_idx]

    # Get effort values for the window
    effort_values = @view contest.efforts[agent_idx, window]
    
    # Hard-code the uniform distribution on domain (lb, ub)
    # initial_pdf(y) = lb ≤ y ≤ ub ? 1. / (ub - lb) : 0.
    # log_init_pdf = log(1. / (ub - lb))

    f(y) = logdomain_integrand(y, effort_values, l)
    
    # Compute the integral of f on domain [lb, ub] to normalise f
    # Use pre-allocated Float64 buffer (estimator doesn't use ForwardDiff)
    segbuf = contest.workspace.estimator_segbufs[agent_idx]
    M, error = quadgk(f, lb, ub;
                     atol=atol,
                     rtol=rtol,
                     segbuf=segbuf)
    
    # Define the normalized estimator μ
    μ(y) = lb ≤ y ≤ ub ? f(y) / M : 0.
    return μ
end



"""
    logdomain_integrand(y, x_vals, l)

Evaluates the log-domain form of:
    f(y) = y^l / ∏(x_i + y)
as:
    f(y) = exp( l*log(y) - sum(log(x_i + y)) )
"""
function logdomain_integrand(y, x_vals::AbstractVector{<:Real}, l::Int)
    @assert y ≥ 0  "y out of bounds"
    acc = zero(y)
    @inbounds @simd for i in eachindex(x_vals)
        acc += log(x_vals[i] + y)
    end
    return exp(l * log(y) - acc)
end