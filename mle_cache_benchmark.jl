#!/usr/bin/env julia
"""
MLE Agents Cache Benchmark

Compares actual performance of the three caching strategies on homogeneous MLE agents
based on the examples/homogeneous-linear-MLE-agents.jl example.

This benchmark tests whether approximate caching is also fastest for MLE agents,
which would justify making caching=:approximate the default setting.
"""

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf
using Random

println("="^60)
println("MLE AGENTS ROOT FINDING CACHE BENCHMARK")
println("="^60)

# Set random seed for reproducibility
Random.seed!(12345)

# Test configuration based on MLE example
const n_agents = 10    # From MLE example
const n_rounds = 1000  # Reduced from 5000 for faster testing
const χ = 0.05
const accuracy = :default  # MLE example uses default accuracy

linear_cost(x) = x  # Linear cost function from MLE example

"""
Generate MLE contest with specified caching strategy
"""
function create_mle_contest(caching::Symbol, cache_tolerance::Float64=1e-9)
    agents = [MLEAgent(linear_cost; χ=χ) for _ in 1:n_agents]
    initial_efforts = [rand() for _ in 1:n_agents]  # Random as in original
    
    return TullockContest(agents, initial_efforts, n_rounds; 
                         accuracy=accuracy,
                         caching=caching, 
                         cache_tolerance=cache_tolerance)
end

"""
Benchmark a single caching strategy for MLE agents
"""
function test_mle_caching_strategy(caching::Symbol, cache_tolerance::Float64=1e-9)
    println("Testing $caching caching...")
    
    # Single timing run
    contest = create_mle_contest(caching, cache_tolerance)
    start_time = time()
    final_round = run!(contest)
    runtime = time() - start_time
    
    # Memory benchmark
    memory_result = @benchmark run!(c) samples=3 evals=1 setup=(
        c = create_mle_contest($caching, $cache_tolerance)
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
    println("  Agent type: MLEAgent") 
    println("  Agents: $n_agents")
    println("  Rounds: $n_rounds") 
    println("  χ: $χ")
    println("  Accuracy: $accuracy")
    println("  Cost function: linear_cost(x) = x")
    println()
    
    # Test all three strategies
    results = []
    
    # 1. No caching (baseline)
    push!(results, test_mle_caching_strategy(:none))
    
    # 2. Exact caching
    push!(results, test_mle_caching_strategy(:exact))
    
    # 3. Test a few approximate tolerances
    tolerances = [1e-12, 1e-9, 1e-6, 1e-3]
    approx_results = []
    
    println("Testing approximate caching tolerances...")
    for tol in tolerances
        result = test_mle_caching_strategy(:approximate, tol)
        push!(approx_results, result)
        @printf("  Tolerance %.0e: %.3f s\n", tol, result.runtime)
    end
    
    # Pick best approximate result
    best_approx = approx_results[argmin([r.runtime for r in approx_results])]
    push!(results, best_approx)
    
    println("\n" * "="^60)
    println("MLE AGENTS RESULTS TABLE")
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
    println("\nMLE AGENTS SUMMARY:")
    exact_speedup = baseline_time / results[2].runtime
    approx_speedup = baseline_time / best_approx.runtime
    
    @printf("• Exact caching: %.2fx speedup\n", exact_speedup)
    @printf("• Best approximate caching: %.2fx speedup (tolerance: %.0e)\n", 
            approx_speedup, best_approx.cache_tolerance)
    
    if exact_speedup > approx_speedup * 1.05
        println("→ MLE Recommendation: Use exact caching")
        mle_winner = :exact
    elseif approx_speedup > exact_speedup * 1.05  
        println("→ MLE Recommendation: Use approximate caching with tolerance $(best_approx.cache_tolerance)")
        mle_winner = :approximate
    else
        println("→ Both caching strategies perform similarly for MLE agents")
        mle_winner = :tie
    end
    
    # Return results for comparison with Bayesian benchmark
    return (
        agent_type = :MLE,
        baseline_time = baseline_time,
        exact_speedup = exact_speedup,
        approx_speedup = approx_speedup,
        best_approx_tolerance = best_approx.cache_tolerance,
        winner = mle_winner,
        results = results
    )
end

# For combined analysis with Bayesian results
function compare_with_bayesian_results(mle_results)
    println("\n" * "="^60)
    println("CROSS-AGENT COMPARISON")
    println("="^60)
    
    # Load previous Bayesian results (approximated from earlier benchmark)
    bayesian_results = (
        agent_type = :Bayesian,
        exact_speedup = 1.69,      # From previous benchmark
        approx_speedup = 1.83,     # From previous benchmark  
        best_approx_tolerance = 1e-9,
        winner = :approximate
    )
    
    println("\nSpeedup Comparison:")
    println("┌─────────────────┬──────────────┬──────────────┬─────────────┐")
    println("│   Agent Type    │ Exact Cache  │ Approx Cache │  Winner     │")
    println("│                 │   Speedup    │   Speedup    │             │")
    println("├─────────────────┼──────────────┼──────────────┼─────────────┤")
    
    @printf("│ %-15s │ %10.2fx   │ %10.2fx   │ %-11s │\n",
            "Bayesian", bayesian_results.exact_speedup, bayesian_results.approx_speedup, 
            string(bayesian_results.winner))
    
    @printf("│ %-15s │ %10.2fx   │ %10.2fx   │ %-11s │\n",
            "MLE", mle_results.exact_speedup, mle_results.approx_speedup,
            string(mle_results.winner))
    
    println("└─────────────────┴──────────────┴──────────────┴─────────────┘")
    
    # Overall recommendation
    println("\nOVERALL RECOMMENDATION:")
    
    bayesian_prefers_approx = (bayesian_results.winner == :approximate)
    mle_prefers_approx = (mle_results.winner == :approximate)
    
    if bayesian_prefers_approx && mle_prefers_approx
        println("✓ RECOMMENDATION: Set caching=:approximate as default")
        println("  Both Bayesian and MLE agents benefit most from approximate caching")
        @printf("  Suggested default tolerance: %.0e\\n", mle_results.best_approx_tolerance)
        recommendation = :set_approximate_default
    elseif bayesian_prefers_approx && (!mle_prefers_approx)
        println("• Mixed results: Bayesian prefers approximate, MLE prefers exact")
        println("  Keep caching=:none as default, let users choose based on agent mix")
        recommendation = :keep_none_default
    elseif (!bayesian_prefers_approx) && mle_prefers_approx
        println("• Mixed results: MLE prefers approximate, Bayesian prefers exact")
        println("  Keep caching=:none as default, let users choose based on agent mix")
        recommendation = :keep_none_default
    else
        println("• Both agent types prefer exact caching")
        println("  Consider setting caching=:exact as default")
        recommendation = :set_exact_default
    end
    
    return recommendation
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    mle_results = main()
    overall_recommendation = compare_with_bayesian_results(mle_results)
    
    # Implementation guidance
    println("\n" * "="^60)
    println("IMPLEMENTATION GUIDANCE")
    println("="^60)
    
    if overall_recommendation == :set_approximate_default
        println("\nTo implement the recommendation:")
        println("1. Change default caching from :none to :approximate in TullockContest constructor")
        println("2. Users can override with caching=:none or caching=:exact if needed")
        println("3. Provides automatic performance boost for both agent types")
    elseif overall_recommendation == :set_exact_default
        println("\nTo implement the recommendation:")
        println("1. Change default caching from :none to :exact in TullockContest constructor")
        println("2. Simpler than approximate caching, no tolerance tuning needed")
        println("3. Provides consistent performance boost across agent types")
    else
        println("\nCurrent implementation is appropriate:")
        println("1. Keep caching=:none as default to avoid unexpected behavior changes")
        println("2. Users can explicitly enable caching based on their agent types")
        println("3. Documentation should highlight performance benefits of caching")
    end
end