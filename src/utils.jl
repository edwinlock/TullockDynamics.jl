"""
Implement binary search root-finding method for function `f` on interval [l, ∞).
This method is quite slow but should be robust.

Assumptions: Function f is strictly decreasing and f(l) ≥ 0.
"""
function find_root(f::Function, l::Float64)::Float64
    @assert f(l) ≥ 0 "Function must satisfy f(l) ≥ 0."
    lower = l
    upper = max(l, 2.0)
    # Phase 1: find appropriate upper bound by repeatedly squaring
    while f(upper) > 0; upper *= upper; end
    # Phase 2: binary search between upper and lower bound
    while upper - lower > 2^(-32)  # TODO: decide on final magic number
        mid = (upper + lower) / 2
        f(mid) == 0 && return mid
        f(mid) > 0 && (lower = mid)
        f(mid) < 0 && (upper = mid)
    end
    mid = (upper + lower) / 2
    @assert mid ≥ l  "Something went wrong with the binary search."
    return mid
end


# For printing contests nicely on the REPL
function Base.show(io::IO, mime::MIME"text/plain", contest::TullockContest)
    print(io, "TullockContest")
    print(io, "\n  ")
    print(io, "Number of agents: $(length(contest.agents))")
    print(io, "\n  ")
    print(io, "Efforts: $(contest.efforts)")
    print(io, "\n  ")
    print(io, "Winners: $(contest.winners)")
    print(io, "\n  ")
    print(io, "Utilities: $(contest.utilities)")
end


# For getting final efforts
final_efforts(contest::TullockContest) = contest.efforts[:, end]


"""
Plot the trajectory of a Tullock contest.
"""
function visualise(contest::TullockContest; rounds=(1,size(contest.efforts)[2]), ylims=:auto)
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
Plot given `data`` matrix for a Tullock contest.
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