"""
# Tullock Contest Dynamics Implementation

This module implements the core dynamics of Tullock contest simulation with the following process:

## Contest Dynamics Process
1. **Initialization**: Each agent starts with a specified initial effort level
2. **Round Structure**: In each round, agents decide whether to update their efforts
3. **Probabilistic Updates**: Each agent flips a biased coin (probability p(t)) to decide on participation
4. **Learning and Adaptation**: If updating, agents:
   - Estimate opponents' total effort using their learning algorithm
   - Compute their best response to this estimate
   - Update their effort using a weighted combination of best response and previous effort
5. **Winner Selection**: A winner is chosen probabilistically based on effort shares
6. **Convergence**: Process continues until Nash gap drops below threshold ε or maximum rounds reached

## Performance Features
- **Threading Support**: Parallel agent updates for improved performance
- **Progress Tracking**: Optional progress bars for long simulations
- **Configurable Precision**: Adjustable numerical integration tolerances
- **Memory Optimization**: Pre-allocated buffers eliminate runtime allocations

## Convergence Behavior
- Nash gaps are not monotonically decreasing
- Convergence occurs when total Nash gap first drops below ε
- Non-convergence is indicated by returning T+1 where T is max rounds
"""

import Random
using StatsBase
using ProgressMeter


"""
    set_efforts!(contest::TullockContest, t::Int; threading::Bool=true, atol::Float64=1e-8, reltol::Float64=1e-6)

Update effort levels for all agents in round `t` of the contest.

This function orchestrates the effort updating process for all agents, with support
for parallel execution and configurable numerical precision for integration.

# Arguments
- `contest::TullockContest`: The contest being simulated
- `t::Int`: Current round number (must be ≥ 2)
- `threading::Bool=true`: Enable parallel processing of agent updates
- `atol::Float64=1e-8`: Absolute tolerance for numerical integration
- `reltol::Float64=1e-6`: Relative tolerance for numerical integration

# Process
For each agent:
1. Probabilistic participation decision based on p(t)
2. If not participating: copy previous round's effort
3. If participating: estimate opponents, compute best response, adapt effort

# Performance
- With `threading=true`: Agents update in parallel using `Threads.@threads`
- With `threading=false`: Sequential updates (useful for debugging or deterministic execution)
- Integration tolerances affect only Bayesian agents using numerical integration

# Requirements
- `t ≥ 2` (agents need at least one round of history)
- Contest must be properly initialized with workspace
"""
function set_efforts!(contest::TullockContest, t::Int; threading::Bool=true, atol::Float64=1e-8, reltol::Float64=1e-6)
    @assert t ≥ 2 "In order to update efforts, the round must be t ≥ 2."

    # Run through all agents and set their efforts in round t
    if threading
        Threads.@threads for i ∈ eachindex(contest.agents)
            _update_agent_effort!(contest, i, t, atol, reltol)
        end
    else
        for i ∈ eachindex(contest.agents)
            _update_agent_effort!(contest, i, t, atol, reltol)
        end
    end
    return nothing
end

"""
    _update_agent_effort!(contest::TullockContest, i::Int, t::Int, atol::Float64, reltol::Float64)

Update effort for a single agent in round `t`.

This internal helper function handles the complete effort update process for one agent,
including probabilistic participation, learning, and adaptation.

# Arguments
- `contest::TullockContest`: The contest being simulated
- `i::Int`: Agent index
- `t::Int`: Current round number
- `atol::Float64`: Absolute tolerance for numerical integration
- `reltol::Float64`: Relative tolerance for numerical integration

# Algorithm
1. **Participation Decision**: Sample from Bernoulli(p(t)) to decide if agent updates
2. **Static Case**: If not updating, copy effort from round t-1
3. **Dynamic Case**: If updating:
   - Extract memory window using h(t)
   - Call agent's estimator to get opponent effort estimate
   - Compute best response to this estimate
   - Blend best response with previous effort using step size α(t)

# Effort Update Formula
```
x[t] = α(t) * best_response + (1 - α(t)) * x[t-1]
```

Where α(t) ∈ [0,1] controls adaptation speed.
"""
function _update_agent_effort!(contest::TullockContest, i::Int, t::Int, atol::Float64, reltol::Float64)
    agent = contest.agents[i]
    # Flip biased coin to determine whether agent updates their effort
    coin = rand()
    if coin >= agent.p(t)  # repeat same effort as in previous round
        contest.efforts[i,t] = contest.efforts[i, t-1]
    else 
        # Get memory window for estimator
        mem_window = agent.h(t)  # get memory window as list or range
        # Determine estimate of total effort of others
        estim = agent.estimator(contest, i, mem_window, atol, reltol)
        # Agent makes their move
        br = best_response(agent, estim, contest.workspace, i, atol, reltol)
        # Compute and store new effort
        prev_effort = contest.efforts[i, t-1]
        x = agent.α(t) * br + (1-agent.α(t)) * prev_effort
        contest.efforts[i,t] = x
    end
end


"""
    update_workspace!(contest::TullockContest, t::Int)

Update workspace buffers after efforts have been set for round `t`.

This function maintains the workspace state by computing derived quantities needed
for utility calculations and subsequent rounds. Must be called after `set_efforts!`
but before utility and Nash gap computations.

# Arguments
- `contest::TullockContest`: Contest with updated efforts for round t
- `t::Int`: Current round number

# Updates
The workspace is updated with:
- `other_efforts[i,t]`: Total effort of all agents except agent i
- `total_efforts[t]`: Sum of all agents' efforts in round t
- `current_round`: Tracks current simulation round

# Performance
- Uses pre-allocated buffers to avoid memory allocations
- Optimized loop structure for cache efficiency
- Required for subsequent utility and Nash gap calculations

# Assumptions
- Efforts for round t have been set via `set_efforts!`
- Workspace buffers are properly sized for the contest
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
    set_utilities!(contest::TullockContest, t::Int)

Compute and store utility values for all agents in round `t`.

Utilities represent each agent's net payoff from their effort choice given all
other agents' efforts. This function assumes efforts and workspace have been
properly updated for the current round.

# Arguments
- `contest::TullockContest`: Contest with updated efforts and workspace
- `t::Int`: Current round number

# Utility Formula
```
utility[i,t] = effort[i,t] / (effort[i,t] + other_efforts[i,t]) - cost[i](effort[i,t])
```

# Components
- **Winning probability**: effort[i,t] / total_effort[t] (effort share)
- **Cost of effort**: agent's cost function applied to their effort
- **Net utility**: Expected payoff minus effort cost

# Requirements
- Efforts must be set for round t
- Workspace must be updated (other_efforts computed)
- Workspace.current_round must equal t

# Storage
Results are stored in `contest.utilities[:,t]` for subsequent analysis.
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
    set_nash_gap!(contest::TullockContest, t::Int)

Compute and store Nash gaps for all agents in round `t`.

The Nash gap measures how much each agent could improve their utility by
unilaterally changing their effort to their best response. A total Nash gap
of zero indicates Nash equilibrium.

# Arguments
- `contest::TullockContest`: Contest with updated efforts and workspace
- `t::Int`: Current round number

# Nash Gap Formula
For each agent i:
```
nash_gap[i,t] = utility(best_response[i], other_efforts[i,t]) - utility(effort[i,t], other_efforts[i,t])
```

# Interpretation
- `nash_gap[i,t] = 0`: Agent i is playing optimally
- `nash_gap[i,t] > 0`: Agent i could improve by changing effort
- `sum(nash_gaps[:,t]) = 0`: System is at Nash equilibrium

# Requirements
- Efforts must be set for round t
- Workspace must be updated (other_efforts computed)
- Workspace.current_round must equal t

# Storage
Results are stored in `contest.nash_gaps[:,t]` for convergence analysis.
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
    step!(contest::TullockContest, t::Int; threading::Bool=true, atol::Float64=1e-8, reltol::Float64=1e-6)

Execute a complete round of the Tullock contest simulation.

This function orchestrates all components of a single round: effort updates,
workspace maintenance, utility calculations, Nash gap computation, and winner selection.

# Arguments
- `contest::TullockContest`: Contest to advance
- `t::Int`: Current round number (1 ≤ t ≤ T)
- `threading::Bool=true`: Enable parallel agent updates
- `atol::Float64=1e-8`: Absolute tolerance for numerical integration
- `reltol::Float64=1e-6`: Relative tolerance for numerical integration

# Round Sequence
1. **Effort Updates** (if t ≥ 2): Agents probabilistically update efforts
2. **Workspace Update**: Compute derived quantities (other_efforts, total_efforts)
3. **Utility Calculation**: Compute net payoffs for all agents
4. **Nash Gap Computation**: Measure distance from equilibrium
5. **Winner Selection**: Randomly select winner based on effort shares

# Winner Selection
Winner chosen using weighted sampling where weights are effort levels:
```
P(agent i wins) = effort[i] / sum(all efforts)
```

# Special Cases
- Round 1: Only workspace update, utilities, Nash gaps, and winner selection
- Rounds ≥ 2: Full sequence including effort updates

# Performance
- Threading affects only the effort update phase
- Integration tolerances affect only Bayesian agents
"""
function step!(contest::TullockContest, t::Int; threading::Bool=true, atol::Float64=1e-8, reltol::Float64=1e-6)
    workspace = contest.workspace

    # Let all agents set their efforts if t ≥ 2
    t ≥ 2 && set_efforts!(contest, t; threading=threading, atol=atol, reltol=reltol)

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
    run!(contest::TullockContest; ε=-1.0, showprogress=false, threading::Bool=true, accuracy::Symbol=:relaxed) -> Int

Execute the complete Tullock contest simulation until convergence or maximum rounds.

This is the main simulation function that runs the contest dynamics, monitoring for
convergence based on the Nash gap and providing optional progress feedback.

# Arguments
- `contest::TullockContest`: Initialized contest to simulate
- `ε::Float64=-1.0`: Nash gap convergence threshold (negative = run full T rounds)
- `showprogress::Bool=false`: Display progress bar during simulation
- `threading::Bool=true`: Enable parallel agent updates
- `accuracy::Symbol=:relaxed`: Integration precision level

# Accuracy Levels
- `:strict`: Ultra-high precision (atol=1e-14, reltol=1e-12) - slowest, maximum accuracy
- `:default`: High precision (atol=1e-10, reltol=1e-8) - very accurate
- `:relaxed`: Balanced precision (atol=1e-8, reltol=1e-6) - good speed/accuracy tradeoff
- `:veryrelaxed`: Lower precision (atol=1e-5, reltol=1e-3) - fastest, small accuracy loss

# Returns
- `Int`: Round number where simulation terminated
  - `1 ≤ result ≤ T`: Converged at this round (Nash gap ≤ ε)
  - `T+1`: Failed to converge within maximum rounds

# Convergence Behavior
- **Nash gap**: Total across all agents: `sum(nash_gaps[:,t])`
- **Non-monotonic**: Nash gap may increase in some rounds
- **First-hit**: Terminates on first round where gap ≤ ε
- **Early termination**: Can converge before reaching T rounds

# Progress Tracking
- **Convergence mode** (ε > 0): Shows Nash gap threshold progress
- **Fixed rounds mode** (ε ≤ 0): Shows round-by-round progress
- Progress bars use ProgressMeter.jl for consistent formatting

# Data Access Warning
If return value > T, accessing `contest.efforts[:,return_value]` will error.
Always use `min(return_value, num_rounds(contest))` when indexing contest data.

# Performance Notes
- Threading primarily affects effort update phase
- Progress bars add minimal overhead
- Integration accuracy affects only Bayesian agents
- Higher accuracy settings significantly slower for Bayesian learning

# Example
```julia
# Run until convergence with progress tracking
final_round = run!(contest, ε=1e-6, showprogress=true, accuracy=:default)

# Run fixed 100 rounds with maximum speed
final_round = run!(contest, threading=true, accuracy=:veryrelaxed)
```
"""
function run!(contest::TullockContest; ε=-1.0, showprogress=false, threading::Bool=true, accuracy::Symbol=:relaxed)
    T = num_rounds(contest)
    t = 1
    
    # Initialize progress bar
    if ε > 0
        prog = ProgressThresh(ε; desc="Nash gap:", enabled=showprogress)
    else
        prog = Progress(T; desc="Round:", enabled=showprogress)
    end
    
    # Get tolerances from accuracy setting
    atol, reltol = accuracy_to_tolerances(accuracy)
    
    while t ≤ T
        step!(contest, t; threading=threading, atol=atol, reltol=reltol)
        gap = nash_gap(contest, t)
        
        # Update progress bar
        if showprogress
            if ε > 0
                update!(prog, gap)
            else
                next!(prog)
            end
        end
        
        # Check convergence
        gap ≤ ε && break
        t += 1
    end
    
    # Finish progress bar
    showprogress && finish!(prog)
    
    return t
end