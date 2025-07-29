#!/usr/bin/env julia
"""
Simple Bayesian Agents Cache Benchmark

Compares actual performance of the three caching strategies on homogeneous Bayesian agents.
"""

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf
using Random

println("="^60)
println("BAYESIAN AGENTS ROOT FINDING CACHE BENCHMARK")
println("="^60)

# Set random seed for reproducibility
Random.seed!(12345)

# Test configuration
const n_agents = 5
const n_rounds = 300  # Reduced for faster testing
const χ = 0.05
const accuracy = :veryrelaxed

cost(x) = x  # Linear cost function

"""
Generate contest with specified caching strategy
"""
function create_test_contest(caching::Symbol, cache_tolerance::Float64=1e-12)
    agents = [BayesianAgent(cost; χ=χ) for _ in 1:n_agents]
    initial_efforts = [rand() for _ in 1:n_agents]
    
    return TullockContest(agents, initial_efforts, n_rounds; 
                         accuracy=accuracy,
                         caching=caching, 
                         cache_tolerance=cache_tolerance)
end

"""
Benchmark a single caching strategy
"""
function test_caching_strategy(caching::Symbol, cache_tolerance::Float64=1e-12)
    println("Testing $caching caching...")
    
    # Single timing run
    contest = create_test_contest(caching, cache_tolerance)
    start_time = time()
    final_round = run!(contest)
    runtime = time() - start_time
    
    # Memory benchmark
    memory_result = @benchmark run!(c) samples=3 evals=1 setup=(
        c = create_test_contest($caching, $cache_tolerance)
    )
    
    # Extract results
    final_efforts = TullockDynamics.final_efforts(contest)
    
    return (
        caching = caching,
        cache_tolerance = cache_tolerance,
        runtime = runtime,
        memory_time = median(memory_result.times) / 1e9,
        memory = median(memory_result.memory),
        allocations = median(memory_result.allocs),
        final_round = final_round,
        final_efforts = final_efforts,
        effort_mean = mean(final_efforts),
        effort_std = std(final_efforts)
    )
end

function main()
    println("Configuration:")
    println("  Agents: $n_agents")
    println("  Rounds: $n_rounds") 
    println("  χ: $χ")
    println("  Accuracy: $accuracy")
    println()
    
    # Test all three strategies
    results = []
    
    # 1. No caching (baseline)
    push!(results, test_caching_strategy(:none))
    
    # 2. Exact caching
    push!(results, test_caching_strategy(:exact))
    
    # 3. Test a few approximate tolerances
    tolerances = [1e-12, 1e-9, 1e-6]
    approx_results = []
    
    println("Testing approximate caching tolerances...")
    for tol in tolerances
        result = test_caching_strategy(:approximate, tol)
        push!(approx_results, result)
        @printf("  Tolerance %.0e: %.3f s\n", tol, result.runtime)
    end
    
    # Pick best approximate result
    best_approx = approx_results[argmin([r.runtime for r in approx_results])]
    push!(results, best_approx)
    
    println("\n" * "="^60)
    println("RESULTS TABLE")
    println("="^60)
    
    # Performance table
    println("\n┌─────────────────┬──────────────┬──────────────┬─────────────┐")
    println("│   Caching       │   Runtime    │   Speedup    │   Memory    │")
    println("│   Strategy      │     (s)      │   vs None    │    (MB)     │")
    println("├─────────────────┼──────────────┼──────────────┼─────────────┤")
    
    baseline_time = results[1].runtime  # :none result
    
    for result in results
        speedup = baseline_time / result.runtime
        memory_mb = result.memory / 1e6
        
        strategy_name = result.caching == :approximate ? 
            "$(result.caching)($(result.cache_tolerance))" : string(result.caching)
        
        @printf("│ %-15s │ %10.3f   │ %10.2fx   │ %9.2f   │\n",
                strategy_name, result.runtime, speedup, memory_mb)
    end
    
    println("└─────────────────┴──────────────┴──────────────┴─────────────┘")
    
    # Convergence comparison
    println("\nConvergence Comparison:")
    println("┌─────────────────┬─────────────┬─────────────┬─────────────┐")
    println("│   Caching       │ Final Round │ Mean Effort │  Std Effort │")
    println("│   Strategy      │             │             │             │")
    println("├─────────────────┼─────────────┼─────────────┼─────────────┤")
    
    for result in results
        strategy_name = result.caching == :approximate ? 
            "$(result.caching)($(result.cache_tolerance))" : string(result.caching)
        
        @printf("│ %-15s │ %9d   │ %9.6f   │ %9.6f   │\n",
                strategy_name, result.final_round, result.effort_mean, result.effort_std)
    end
    
    println("└─────────────────┴─────────────┴─────────────┴─────────────┘")
    
    # Summary
    println("\nSUMMARY:")
    exact_speedup = baseline_time / results[2].runtime
    approx_speedup = baseline_time / best_approx.runtime
    
    @printf("• Exact caching: %.2fx speedup\n", exact_speedup)
    @printf("• Best approximate caching: %.2fx speedup (tolerance: %.0e)\n", 
            approx_speedup, best_approx.cache_tolerance)
    
    if exact_speedup > approx_speedup * 1.05
        println("→ Recommendation: Use exact caching")
    elseif approx_speedup > exact_speedup * 1.05  
        println("→ Recommendation: Use approximate caching with tolerance $(best_approx.cache_tolerance)")
    else
        println("→ Both caching strategies perform similarly")
    end
    
    return results
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end