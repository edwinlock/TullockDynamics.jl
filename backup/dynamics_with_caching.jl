"""
Implementation of the following Tullock dynamics:
1. Each agent starts with an initial effort.
2. In each round, the agents each flip a (biased) coin to decide whether to update their effort.
3. If an agent decides to update, they estimate the total effort of their opponents, and best respond to that by setting a new effort.
4. The dynamics runs until the maximum number of rounds is reached or the Nash gap is sufficiently small.
"""

import Random
using StatsBase

"""
Cached wrapper for best_response calls with round-based invalidation.
Main benefit: when no agents update in a round, we can reuse computations.
"""
function cached_best_response(agent::Agent, estim, round::Int; min_other_efforts, max_other_efforts, _ignore...)
    # Update statistics
    CACHE_STATS.total_calls += 1
    
    # Create stable cache key based on agent and estimate
    if estim isa Function
        # For PDF estimates, use a generation number instead of sampling
        # This requires tracking when PDFs change (implemented in estimators)
        estim_key = (hash(agent.estimator), round)  # Round acts as generation
    else
        # For scalar estimates, use the value directly
        estim_key = estim
    end
    
    cache_key = (
        hash(agent.cost),       # Cost function identity
        agent.χ,                # Minimum effort bound
        estim_key,              # Estimate identifier
        min_other_efforts,      # Bounds
        max_other_efforts,
        round                   # Round for invalidation
    )
    
    # Check cache first
    cached_result = get(BEST_RESPONSE_CACHE, cache_key, nothing)
    if cached_result !== nothing
        CACHE_STATS.hits += 1
        return cached_result
    end
    
    # Cache miss - compute best response and cache it
    CACHE_STATS.misses += 1
    result = best_response(agent, estim; 
                         min_other_efforts=min_other_efforts, 
                         max_other_efforts=max_other_efforts)
    BEST_RESPONSE_CACHE[cache_key] = result
    return result
end


"""Set efforts of all agents in round `t` of TC `contest`."""
function set_efforts!(contest::TullockContest, t::Int)
    @assert t ≥ 2 "In order to update efforts, the round must be t ≥ 2."

    # Run through all agents and set their efforts in round t
    for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        # Flip biased coin to determine whether agent updates their effort
        coin = rand()
        if coin >= agent.p(t)  # repeat same effort as in previous round
            contest.efforts[i,t] = contest.efforts[i, t-1]
        else 
            # Get memory window for estimator
            mem_window = agent.h(t)  # get memory window as list or range
            # Determine estimate of total effort of others (no caching - ineffective)
            estim = agent.estimator(contest, i, mem_window)
            # Agent makes their move (cached with round-based invalidation)
            br = cached_best_response(agent, estim, t;
                min_other_efforts=contest.workspace.min_other_bounds[i],
                max_other_efforts=contest.workspace.max_other_bounds[i],
            )
            # Compute and store new effort
            prev_effort = contest.efforts[i, t-1]
            x = agent.α(t) * br + (1-agent.α(t)) * prev_effort
            contest.efforts[i,t] = x
        end
    end
    return nothing
end


"""
Update workspace in round t. Assumes that efforts have already been set.

This function updates five variables:
    own_efforts
    other_efforts
    total_effort
    weights_obj
    current_round
"""

function update_workspace!(contest::TullockContest, t::Int)
    ws = contest.workspace

    # Compute total_effort and own_efforts
    ws.total_effort = 0.0  # reset variable
    for i ∈ eachindex(contest.agents)
        own_effort = contest.efforts[i,t]
        ws.own_efforts[i] = own_effort
        ws.total_effort += own_effort
    end

    # Compute other_efforts
    for i ∈ eachindex(contest.agents)
        ws.other_efforts[i] = ws.total_effort - ws.own_efforts[i]
    end

    # Update weights
    ws.weights_obj.values .= ws.own_efforts
    
    # Mark workspace as updated for this round
    ws.current_round = t

    return nothing
end

"""
Compute utilities of all agents in round `t` of TC `contest'.

If workspace is not updated for this round, this function will update it first.
"""
function set_utilities!(contest::TullockContest, t::Int)
    ws = contest.workspace
    
    # Check if workspace needs updating for this round
    if ws.current_round != t
        update_workspace!(contest, t)
    end

    # Compute and store utility for each agent in round t
    for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        contest.utilities[i,t] = utility(agent, ws.own_efforts[i], ws.other_efforts[i])
    end

    return nothing
end


"""
Compute Nash gap of all agents in round `t` of TC `contest'.

If workspace is not updated for this round, this function will update it first.
"""
function set_nash_gap!(contest::TullockContest, t::Int)
    ws = contest.workspace
    
    # Check if workspace needs updating for this round
    if ws.current_round != t
        update_workspace!(contest, t)
    end

    for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        contest.nash_gaps[i,t] = nash_gap(agent, ws.own_efforts[i], ws.other_efforts[i])
    end
    
    return nothing
end


"""
Run round `t` of TC `contest`. This involves getting agents to update their
efforts, and then determining a winner.
"""
function step!(contest::TullockContest, t::Int)
    workspace = contest.workspace

    # Let all agents set their efforts if t ≥ 2
    t ≥ 2 && set_efforts!(contest, t)

    # Update the workspace now that efforts have been set
    update_workspace!(contest, t)

    # Compute the utilities of all the agents
    set_utilities!(contest, t)
    
    # Compute the Nash gap of each agent
    set_nash_gap!(contest, t)
    
    # Determine and record the winner
    winner = sample(workspace.weights_obj)
    contest.winners[winner, t] = true

    return nothing
end


"""
Run the contest until the maximum number of rounds is reached or the Nash gap drops to
≤ ε. Default ε is set to -1.0 so the dynamics runs for maximum number of rounds.

Returns:
- If converged early: the round number where convergence occurred (1 ≤ result ≤ T)
- If failed to converge: T+1 (to signal non-convergence)

Note: the Nash gap is not monotonically decreasing, but the dynamics terminates when
the gap drops below ε for the first time.

WARNING: The returned value may exceed the allocated matrix size if convergence failed.
When accessing contest data, always use min(returned_round, num_rounds(contest)) or 
use the final_efforts() function which safely accesses the last column.
"""
function run!(contest::TullockContest; ε=-1.0, showprogress=false)
    T = num_rounds(contest)
    t = 1
    prog = ProgressThresh(ε; desc="Minimizing:", enabled=showprogress)
    while t ≤ T
        step!(contest, t)
        gap = nash_gap(contest, t)
        gap ≤ ε && break
        t += 1
        update!(prog, gap)
    end
    return t
end