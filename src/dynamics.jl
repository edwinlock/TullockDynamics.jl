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
    min_total_efforts = sum(agent.χ for agent in contest.agents)
    max_total_efforts = sum(agent.max_effort for agent in contest.agents)
    @assert t ≥ 2 "In order to update efforts, the round must be t ≥ 2."
    # Run through all agents and set their efforts in round t
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        # Flip biased coin to determine whether agent updates their effort
        coin = rand()
        if coin >= agent.p(t)  # repeat same effort as in previous round
            contest.efforts[i,t] = contest.efforts[i, t-1]
        else 
            # Retrieve data within memory window for estimator
            mem_window = agent.h(t)  # get memory window as list or range
            own_efforts, total_efforts, wins = retrieve_estimator_data(i, mem_window, contest)
            # Determine estimate
            est = agent.estimator(  # estimate of total effort of others
                own_efforts=own_efforts,
                total_efforts=total_efforts,
                wins=wins,
                cost=agent.cost,
                min_other_efforts=min_total_efforts - agent.χ,
                max_other_efforts=max_total_efforts - agent.max_effort,
            )
            # Agent makes their move
            br = best_response(agent, est;
                min_other_efforts=min_total_efforts - agent.χ,
                max_other_efforts=max_total_efforts - agent.max_effort,
            )
            prev_effort = contest.efforts[i, t-1]
            x = agent.α(t) * br + (1-agent.α(t)) * prev_effort
            contest.efforts[i,t] = x
        end
    end
    return nothing
end


function retrieve_estimator_data(i, mem_window, contest::TullockContest)
    own_efforts = contest.efforts[i, mem_window]
    total_efforts = vec(sum(contest.efforts[:, mem_window]; dims=1))
    wins = contest.winners[i, mem_window]
    return own_efforts, total_efforts, wins
end


"""
Compute utilities of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed.
"""
function set_utilities!(contest::TullockContest, t::Int)
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        all_efforts = contest.efforts[:,t]
        x = all_efforts[i]
        s = sum(all_efforts) - x
        contest.utilities[i,t] = utility(agent, x, s)
    end
    return nothing
end


"""
Compute Nash gap of all agents in round `t` of TC `contest'.

Assumes that efforts have already been computed.
"""
function set_nash_gap!(contest::TullockContest, t::Int)
    Threads.@threads for i ∈ eachindex(contest.agents)
        agent = contest.agents[i]
        all_efforts = contest.efforts[:,t]
        x = all_efforts[i]
        s = sum(all_efforts) - x
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
    # Determine a winner
    latest_efforts = contest.efforts[:, t]  # the effort in round t for each agent
    winner = sample(Weights(latest_efforts))
    contest.winners[winner, t] = true
    return nothing
end


"""
Run the contest until the maximum number of rounds is reached or the Nash gap drops to
≤ ε. Default ε is set to -1.0 so the dynamics runs for maximum number of rounds.

Note: the Nash gap is not monotonically decreasing, but the dynamics terminates when
the gap drops below ε for the first time.
"""
function run!(contest::TullockContest; ε=-1.0, showprogress=false)
    T = num_rounds(contest)
    t = 1
    showprogress && (p = ProgressMeter.Progress(T))
    while t ≤ T
        step!(contest, t)
        nash_gap(contest, t) ≤ ε && break
        t += 1
        showprogress && ProgressMeter.next!(p)
    end
    return t
end
