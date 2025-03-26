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
function Plots.plot(contest::TullockContest; ylims)
    T = num_rounds(contest)
    x = 1:T
    agentlabels = permutedims(["Agent $(i)" for i ∈ 1:T])
    plot(
        x, contest.efforts',
        ylims=ylims,
        xlabel="round",
        ylabel=ylabel,
        # markershape=:circle,
        labels=agentlabels
    )
end




"""
Plot given `data`` matrix for a Tullock contest.
"""
function Plots.plot(data::Matrix; ylims, ylabel)
    T = size(data)[2]
    x = 1:T
    agentlabels = permutedims(["Agent $(i)" for i ∈ 1:T])
    plot(
        x, data',
        ylims=ylims,
        xlabel="round",
        ylabel=ylabel,
        # markershape=:circle,
        labels=agentlabels,
        palette = :darkrainbow,
    )
end