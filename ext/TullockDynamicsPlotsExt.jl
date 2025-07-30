module TullockDynamicsPlotsExt

using TullockDynamics
using Plots
using Measures

"""
    visualise(contest::TullockContest; rounds=(1,size(contest.efforts)[2]), ylims=:auto)

Plot the trajectory of a Tullock contest.

This function creates a comprehensive visualization showing:
- Agent effort trajectories over time
- Agent utility trajectories over time  
- Individual Nash gaps over time
- Total (summed) Nash gap over time

# Arguments
- `contest::TullockContest`: The contest to visualize
- `rounds`: Tuple specifying the range of rounds to plot, defaults to all rounds
- `ylims`: Y-axis limits, defaults to automatic scaling

# Returns
- Combined plot with four subplots showing the contest dynamics

# Example
```julia
using TullockDynamics
using Plots  # Required for plotting functionality

# Create and run a contest
agents = [MLEAgent(x -> x^2), BayesianAgent(x -> x^2)]
contest = TullockContest(agents, [0.1, 0.15], 50)
run!(contest)

# Visualize the results
visualise(contest)

# Plot specific rounds with custom y-limits
visualise(contest; rounds=(10, 40), ylims=(0, 0.5))
```

# Requirements
This function requires Plots.jl to be loaded. The plotting functionality is provided
via a package extension and will only be available when Plots.jl is in the environment.
"""
function TullockDynamics.visualise(contest::TullockDynamics.TullockContest; rounds=(1,size(contest.efforts)[2]), ylims=:auto)
    n, T = size(contest.efforts)
    efforts_plt = visualise(contest.efforts, rounds=rounds, ylims=:auto, ylabel="effort", yscale=:identity)
    util_plt = visualise(contest.utilities, rounds=rounds, ylims=:auto, ylabel="utility", yscale=:identity)
    individual_nash_plt = visualise(contest.nash_gaps, rounds=rounds, ylims=:auto, ylabel="Individual Nash gaps", yscale=:identity)
    # summed Nash gap
    total_nash_gap = vec(sum(contest.nash_gaps; dims=1))
    x = max(rounds[1],1):max(rounds[2],T)
    summed_nash_plt = plot(
        x, total_nash_gap[x],
        ylims=:auto, ylabel="Summed Nash gap", legend=:none)
    # combine all four plots into one, side by side
    l = @layout [ a; [b c d]]
    plot(efforts_plt, util_plt, individual_nash_plt, summed_nash_plt, layout=l, size=(1100,600), margin=5mm)
end

"""
    visualise(data::Matrix; rounds=(1,size(data)[2]), ylims=:auto, ylabel, yscale=:identity)

Plot given `data` matrix for a Tullock contest.

# Arguments
- `data::Matrix`: Data matrix to plot (agents × rounds)
- `rounds`: Tuple specifying the range of rounds to plot
- `ylims`: Y-axis limits
- `ylabel`: Label for the y-axis
- `yscale`: Y-axis scale (`:identity` or `:log10`)

# Returns
- Plot object showing the data trajectories
"""
function visualise(data::Matrix; rounds=(1,size(data)[2]), ylims=:auto, ylabel, yscale=:identity)
    n, T = size(data)
    x = max(rounds[1],1):max(rounds[2],T)
    agentlabels = permutedims(["Agent $(i)" for i ∈ 1:n])
    plot(
        x, data[:, x]',
        ylims=ylims,
        xlabel="round",
        ylabel=ylabel,
        # markershape=:circle,
        labels=agentlabels,
        palette = :darkrainbow,
        yscale=yscale,
    )
end

end