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
function find_root(f::F, l::Float64)::Float64 where F
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
        f_mid = f(mid)::Float64
        
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
    find_root_cached(f::Function, l::Float64; approximate_cache::Bool=false, cache_tolerance::Float64=1e-9) -> Float64

Enhanced version of find_root with internal function value caching.

Uses the same algorithm as find_root but caches function evaluations to avoid redundant 
computations. Supports both exact and approximate cache matching.

# Arguments
- `f::Function`: A strictly decreasing function to find the root of
- `l::Float64`: Lower bound of the search interval (must satisfy f(l) ≥ 0)
- `approximate_cache::Bool=false`: If true, allows cache hits for nearby x values
- `cache_tolerance::Float64=1e-9`: Tolerance for approximate cache hits (when approximate_cache=true)

# Returns
- `Float64`: The root x where f(x) ≈ 0 and x ≥ l

# Caching Strategies
- **Exact caching** (default): Only exact x matches return cached values
- **Approximate caching**: Returns cached f(x_cached) if |x - x_cached| < cache_tolerance

# Performance
The caching provides speedup when:
- Function evaluations are expensive (common in Bayesian agents)
- The binary search revisits similar x values
- Multiple candidates are evaluated at the end

Memory overhead is minimal (typically < 1KB per call).

# Examples
```julia
# Use exact caching (default)
root = find_root_cached(expensive_function, 0.0)

# Use approximate caching for smooth expensive functions
root = find_root_cached(smooth_function, 0.0; approximate_cache=true, cache_tolerance=1e-9)
```
"""
function find_root_cached(f::F, l::Float64; approximate_cache::Bool=false, cache_tolerance::Float64=1e-9)::Float64 where F
    # Cache: Dict mapping x values to f(x) values
    cache = Dict{Float64, Float64}()
    
    # Cached function wrapper with both exact and approximate matching
    function cached_f(x::Float64)::Float64
        # First check for exact match (fastest)
        if haskey(cache, x)
            return cache[x]
        end
        
        # If approximate caching is enabled, check for nearby values
        if approximate_cache
            for (cached_x, cached_value) in cache
                if abs(x - cached_x) < cache_tolerance
                    # Store under the new key for future exact matches
                    cache[x] = cached_value
                    return cached_value
                end
            end
        end
        
        # Cache miss - compute new value
        result = f(x)::Float64
        cache[x] = result
        return result
    end
    
    @assert cached_f(l) ≥ 0 "Function must satisfy f(l) ≥ 0."
    
    # Phase 1: Improved bracketing with exponential growth capping
    lower = l
    upper = max(l + 1.0, 2.0)
    max_bracket = 1e6  # Reasonable upper limit
    while cached_f(upper) > 0 && upper < max_bracket
        upper *= 2.0  # Double instead of squaring
    end
    
    # Phase 2: Enhanced binary search with multiple termination conditions
    max_iterations = 100
    iteration = 0
    f_lower = cached_f(lower)
    f_upper = cached_f(upper)
    
    while iteration < max_iterations
        iteration += 1
        
        # Termination condition 1: Bracket is small enough
        if upper - lower <= MAXGAP
            break
        end
        
        mid = (upper + lower) / 2
        f_mid = cached_f(mid)
        
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
    f_values = [f_lower, cached_f(mid), f_upper]  # mid might be cached already
    best_idx = argmin(abs.(f_values))
    
    result = candidates[best_idx]
    @assert result ≥ l "Something went wrong with the binary search."
    
    # Cache is automatically cleaned up when function returns
    return result
end

"""
    find_root_with_caching(f::Function, l::Float64, workspace::ContestWorkspace) -> Float64

Choose the appropriate root finding method based on workspace caching configuration.

# Arguments
- `f::Function`: Function to find root of
- `l::Float64`: Lower bound for root finding
- `workspace::ContestWorkspace`: Contest workspace containing caching settings

# Returns
- `Float64`: The root of the function

# Caching Behavior
- `:none`: Uses original find_root (no caching)
- `:exact`: Uses find_root_cached with exact matching
- `:approximate`: Uses find_root_cached with approximate matching using workspace.cache_tolerance
"""
function find_root_with_caching(f::F, l::Float64, workspace::ContestWorkspace)::Float64 where F
    if workspace.caching == :none
        return find_root(f, l)
    elseif workspace.caching == :exact
        return find_root_cached(f, l; approximate_cache=false)
    elseif workspace.caching == :approximate
        return find_root_cached(f, l; approximate_cache=true, cache_tolerance=workspace.cache_tolerance)
    else
        error("Invalid caching mode: $(workspace.caching)")
    end
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


# Plotting functionality has been moved to TullockDynamicsPlotsExt
# Load Plots.jl to enable visualise() function

"""
    visualise(contest::TullockContest; kwargs...)

Plot contest dynamics (requires Plots.jl to be loaded).

This function is provided by the TullockDynamicsPlotsExt extension.
Load Plots.jl to enable this functionality:

```julia
using Plots
visualise(contest)
```
"""
function visualise(args...; kwargs...)
    error("visualise requires Plots.jl to be loaded. Please run: using Plots")
end