"""
Implementation of the following Tullock dynamics:
1. Each agent starts with an initial effort.
2. In each round, the agents each flip a (biased) coin to decide whether to update their effort.
3. If an agent decides to update, they estimate the total effort of their opponents, and best respond to that by setting a new effort.
4. The dynamics runs until the maximum number of rounds is reached or the Nash gap is sufficiently small.
"""

import Random
using StatsBase

"""Set efforts of all agents in round `t` of TC `contest`."""
function set_efforts!(contest::TullockContest, t::Int)
    @assert t ≥ 2 "In order to update efforts, the round must be t ≥ 2."
    # Run through all agents and set their efforts in round t
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        # Flip biased coin to determine whether agent updates their effort
        coin = rand()
        if coin >= agent.p(t)  # repeat same effort as in previous round
            contest.efforts[i,t] = contest.efforts[i, t-1]
        else 
            # Get memory window for estimator
            mem_window = agent.h(t)  # get memory window as list or range
            # Determine estimate of total effort of others
            estim = agent.estimator(contest, i, mem_window)
            # Agent makes their move
            br = best_response(agent, estim;
                min_other_efforts=contest.workspace.min_other_bounds[i],
                max_other_efforts=contest.workspace.max_other_bounds[i],
            )
            prev_effort = contest.efforts[i, t-1]
            x = agent.α(t) * br + (1-agent.α(t)) * prev_effort
            contest.efforts[i,t] = x
        end
    end
    return nothing
end


"""
Compute utilities of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed.
"""
function set_utilities!(contest::TullockContest, t::Int)
    workspace = contest.workspace
    # Direct computation without intermediate allocation
    workspace.total_effort = 0.0
    for i in eachindex(workspace.all_efforts)
        workspace.all_efforts[i] = contest.efforts[i, t]
        workspace.total_effort += workspace.all_efforts[i]
    end
    
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        x = workspace.all_efforts[i]
        s = workspace.total_effort - x
        contest.utilities[i,t] = utility(agent, x, s)
    end
    return nothing
end


"""
Compute Nash gap of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed.
"""
function set_nash_gap!(contest::TullockContest, t::Int)
    workspace = contest.workspace
    # Reuse the data already computed in set_utilities! if it's the same round
    # Otherwise recompute (this is a minor optimization for the common case)
    if workspace.total_effort == 0.0  # Not computed yet this round
        workspace.total_effort = 0.0
        for i in eachindex(workspace.all_efforts)
            workspace.all_efforts[i] = contest.efforts[i, t]
            workspace.total_effort += workspace.all_efforts[i]
        end
    end
    
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        x = workspace.all_efforts[i]
        s = workspace.total_effort - x
        contest.nash_gaps[i,t] = nash_gap(agent, x, s)
    end
    return nothing
end


"""
Run round `t` of TC `contest`. This involves getting agents to update their
efforts, and then determining a winner.
"""
function step!(contest::TullockContest, t::Int)
    # Let all agents set their efforts if t ≥ 2
    t ≥ 2 && set_efforts!(contest, t)
    # Compute the utilities of all the agents
    set_utilities!(contest, t::Int)
    # Compute the Nash gap of each agent
    set_nash_gap!(contest, t::Int)
    # Determine a winner using reusable Weights object
    workspace = contest.workspace
    # Update the Weights object in-place (all_efforts buffer already populated)
    workspace.weights_obj.values .= workspace.all_efforts
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

⚠️  WARNING: The returned value may exceed the allocated matrix size if convergence failed.
When accessing contest data, always use min(returned_round, num_rounds(contest)) or 
use the final_efforts() function which safely accesses the last column.

Note: the Nash gap is not monotonically decreasing, but the dynamics terminates when
the gap drops below ε for the first time.
"""
function run!(contest::TullockContest; ε=-1.0, showprogress=false)
    T = num_rounds(contest)
    t = 1
    converged = false
    # prog = Progress(T, enabled=showprogress)
    prog = ProgressThresh(ε; desc="Minimizing:", enabled=showprogress)
    while t ≤ T
        step!(contest, t)
        gap = nash_gap(contest, t)
        if gap ≤ ε
            converged = true
            break
        end
        t += 1
        # ProgressMeter.next!(prog)
        # update!(p, gap)
    end
    # Return the round number, or T+1 if failed to converge (for backward compatibility)
    # But accessing contest data should only use valid indices 1:T
    return converged ? t : T + 1
end