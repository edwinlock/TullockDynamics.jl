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
            br = best_response(agent, estim, contest.workspace, i)
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

This function updates three variables:
    ws.other_efforts[:,t]
    ws.total_effort[t]
    ws.current_round
"""

function update_workspace!(contest::TullockContest, t::Int)
    ws = contest.workspace

    # Compute other_efforts and total_efforts entries for round t
    ws.total_efforts[t] = 0.  # reset variable
    for i ∈ eachindex(contest.agents)
        ws.total_efforts[t] += contest.efforts[i,t]
    end

    # Compute other_efforts
    for i ∈ eachindex(contest.agents)
        ws.other_efforts[i,t] = ws.total_efforts[t] - contest.efforts[i,t]
    end

    # Update round
    ws.current_round = t

    return nothing
end

"""
Compute utilities of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed and the workspace has been updated.
"""
function set_utilities!(contest::TullockContest, t::Int)
    ws = contest.workspace
    @assert ws.current_round == t  "Workspace hasn't been updated"

    # Compute and store utility for each agent in round t
    for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        contest.utilities[i,t] = utility(agent, contest.efforts[i,t], ws.other_efforts[i,t])
    end

    return nothing
end


"""
Compute Nash gap of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed and workspace is updated.
"""
function set_nash_gap!(contest::TullockContest, t::Int)
    ws = contest.workspace
    @assert ws.current_round == t  "Workspace hasn't been updated"

    for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        contest.nash_gaps[i,t] = nash_gap(agent, contest.efforts[i,t], ws.other_efforts[i,t])
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
    winner = sample(Weights(contest.efforts[:,t]))
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
    # prog = Progress(T, enabled=showprogress)
    prog = ProgressThresh(ε; desc="Minimizing:", enabled=showprogress)
    while t ≤ T
        step!(contest, t)
        gap = nash_gap(contest, t)
        gap ≤ ε && break
        t += 1
        # ProgressMeter.next!(prog)
        # update!(p, gap)
    end
    return t
end