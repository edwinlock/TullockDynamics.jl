"""
Implementation of maximum likelihood estimator, given own `efforts` and `wins`.
"""
function max_likelihood_estimator(; own_efforts::Vector{Float64}, wins::Vector{Bool}, _ignore...)::Float64
    w = sum(wins)  # compute number of wins
    # Deal with edge case of zero wins
    w == 0 && return 1  # TODO: discuss with Abheek and Sonja!
    # Solve sum(x[i] / (x[i] + y)) = w for y:
    # Define strictly decreasing function for root finding
    f(y) = sum(x / (x + y) for x in own_efforts) - w
    return find_root(f, 0.0)
    # return find_zero(f, (0.0, Inf))
    # return find_zero(f, 0.0)
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
    # return find_zero(f, (0.0, Inf))
    # return find_zero(f, 0.0)
end


"""
Implementation of "dumb" estimate, given own `efforts' and `wins`.
"""
function dumb_estimator(; own_efforts::Vector{Float64}, wins::Vector{Bool}, _ignore...)::Float64
    w = sum(wins)  # compute number of wins
    # Deal with edge case of zero wins
    w == 0 && return 1  # TODO: discuss with Abheek and Sonja!
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
function classic_estimator(; own_efforts::Vector{Float64}, total_efforts::Vector{Float64}, _ignore...)::Float64
    return total_efforts[end] - own_efforts[end]
end



"""
Implementation of Bayesian estimate, given own `efforts` and `wins`.

TODO: Finish this!
"""
function bayesian_estimator(; own_efforts::Vector{Float64}, wins::Int, _ignore...)
    # TODO: Finish this.
end
