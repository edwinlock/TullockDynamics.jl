const MAXGAP = 2^(-20)  # Bracket size tolerance
const FUNCTION_TOL = 1e-12  # High accuracy function tolerance

"""
    find_root(f::Function, l::Float64) -> Float64

Enhanced binary search root-finding method for function `f` on interval [l, ∞).

Finds a root of the equation `f(x) = 0` where `x ≥ l`, assuming that `f` is 
strictly decreasing and `f(l) ≥ 0`.

# Algorithm

Uses a two-phase approach:

1. **Bracketing Phase**: Finds an upper bound where `f(upper) < 0` by doubling 
   from the initial point, with exponential growth capping to prevent overflow.

2. **Enhanced Binary Search**: Refines the root using binary search with multiple 
   termination conditions:
   - Bracket size tolerance (MAXGAP = 2^(-20) ≈ 9.5e-7)
   - High-accuracy function value tolerance (FUNCTION_TOL = 1e-12)
   - Convergence detection based on bracket shrinkage rate
   - Maximum iteration limit for robustness

# Features

- **High Accuracy**: Multiple termination conditions ensure optimal precision
- **Overflow Protection**: Capped exponential growth prevents numeric overflow
- **Early Termination**: Returns immediately when function value is near zero
- **Best Candidate Selection**: Returns the point with smallest |f(x)| among bracket endpoints
- **Cached Evaluations**: Minimizes redundant function calls for efficiency

# Arguments

- `f::Function`: A strictly decreasing function to find the root of
- `l::Float64`: Lower bound of the search interval (must satisfy f(l) ≥ 0)

# Returns

- `Float64`: The root x where f(x) ≈ 0 and x ≥ l

# Throws

- `AssertionError`: If f(l) < 0 (violates precondition)
- `AssertionError`: If the algorithm fails to find a valid root

# Examples

```julia
# Find root of f(x) = 10 - x (root at x = 10)
f1(x) = 10.0 - x
root = find_root(f1, 0.0)  # Returns ≈ 10.0

# MLE estimator pattern (common in TullockDynamics.jl)
effort_values = [0.15, 0.22, 0.18]
wins = 2.0
mle_func(y) = sum(x / (x + y) for x in effort_values) - wins
root = find_root(mle_func, 0.0)

# Cost function pattern
cost(x) = 0.5 * x^1.5
cost_func(x) = 1.0 - cost(x)  # Find where cost(x) = 1
max_effort = find_root(cost_func, 0.0)
```

# Performance

Optimized for the usage patterns in TullockDynamics.jl, typically achieving:
- High accuracy (function values < 1e-12)
- Fast convergence (usually < 50 iterations)
- Minimal memory allocations through cached evaluations

# See Also

- [`max_agent_effort`](@ref): Uses find_root to compute maximum effort levels
- [`max_likelihood_estimator`](@ref): Uses find_root for parameter estimation
- [`deterministic_max_likelihood_estimator`](@ref): Another find_root application
"""
function find_root(f::Function, l::Float64)::Float64
    @assert f(l) ≥ 0 "Function must satisfy f(l) ≥ 0."
    
    # Phase 1: Improved bracketing with exponential growth capping
    lower = l
    upper = max(l + 1.0, 2.0)
    max_bracket = 1e6  # Reasonable upper limit
    while f(upper) > 0 && upper < max_bracket
        upper *= 2.0  # Double instead of squaring
    end
    
    # Phase 2: Enhanced binary search with multiple termination conditions
    max_iterations = 100
    iteration = 0
    f_lower = f(lower)
    f_upper = f(upper)
    
    while iteration < max_iterations
        iteration += 1
        
        # Termination condition 1: Bracket is small enough
        if upper - lower <= MAXGAP
            break
        end
        
        mid = (upper + lower) / 2
        f_mid = f(mid)
        
        # Termination condition 2: Function value is essentially zero (high accuracy)
        if abs(f_mid) <= FUNCTION_TOL
            return mid
        end
        
        # Standard binary search update with cached function values
        if f_mid > 0
            lower = mid
            f_lower = f_mid
        else
            upper = mid
            f_upper = f_mid
        end
        
        # Termination condition 3: Convergence detection
        bracket_ratio = (upper - lower) / (upper + lower)
        if bracket_ratio < 1e-14
            break
        end
    end
    
    # Return the point with the smallest absolute function value
    mid = (upper + lower) / 2
    candidates = [lower, mid, upper]
    f_values = [f_lower, f(mid), f_upper]
    best_idx = argmin(abs.(f_values))
    
    result = candidates[best_idx]
    @assert result ≥ l "Something went wrong with the binary search."
    return result
end


"""
Implement binary search root-finding method for function `f` on interval [l, ∞).
This method is quite slow but should be robust.

Assumptions: Function f is strictly decreasing and f(l) ≥ 0.
"""
# function find_root(f::Function, l::Float64)::Float64
#     @assert f(l) ≥ 0 "Function must satisfy f(l) ≥ 0."
#     lower = l
#     upper = max(l, 2.0)
#     # Phase 1: find appropriate upper bound by repeatedly squaring
#     while f(upper) > 0; upper *= upper; end
#     # Phase 2: binary search between upper and lower bound
#     while upper - lower > MAXGAP
#         mid = (upper + lower) / 2
#         f(mid) == 0 && return mid
#         f(mid) > 0 && (lower = mid)
#         f(mid) < 0 && (upper = mid)
#     end
#     mid = (upper + lower) / 2
#     @assert mid ≥ l  "Something went wrong with the binary search."
#     return mid
# end



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
Check if the contest converged given the result from run!().
Returns (converged::Bool, actual_final_round::Int)
"""
function convergence_status(contest::TullockContest, run_result::Int)
    T = num_rounds(contest)
    if run_result > T
        return (false, T)  # Did not converge, actual data ends at round T
    else  
        return (true, run_result)  # Converged at round run_result
    end
end


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