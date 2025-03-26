"""
Implementation of the following Tullock dynamics:
1. Each agent starts with an initial effort.
2. In each round, each agent flips a (biased) coin to decide whether to update their effort.
3. If an agent decides to update, they estimate the total effort of their opponents, and best respond to that by setting a new effort.
4. 
"""

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
        else  # 
            # Retrieve data within memory window for estimator
            mem_win = agent.window(t)  # get memory window as list or range
            own_efforts = contest.efforts[i, mem_win]
            total_efforts = vec(sum(contest.efforts[:, mem_win]; dims=1))
            wins = contest.winners[i, mem_win]
            # Determine estimate
            est = agent.estimator(  # estimate of total effort of others
                own_efforts=own_efforts,
                total_efforts=total_efforts,
                wins=wins
            )
            # Agent makes their move
            br = best_response(agent, est)
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
Run the contest.
"""
function run!(contest::TullockContest)
    println("Running contest")
    T = num_rounds(contest)
    @showprogress for t ∈ 1:T
        step!(contest, t)
    end
    return nothing
end
