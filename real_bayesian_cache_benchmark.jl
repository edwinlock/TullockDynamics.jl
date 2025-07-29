#!/usr/bin/env julia
"""
Real Homogeneous Bayesian Agents Cache Benchmark

Compares actual performance of non-cached, exact cached, and approximate cached 
root finding using the real homogeneous-Bayesian-agents.jl example.

This benchmark creates actual Bayesian agent contests with different caching 
strategies and measures:
1. Runtime performance
2. Memory allocation  
3. Final effort convergence
4. Optimal approximate tolerance levels

Author: TullockDynamics.jl Performance Analysis
"""

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf
using Random

println("="^80)
println("REAL HOMOGENEOUS BAYESIAN AGENTS CACHE BENCHMARK")
println("="^80)
println("Comparing root finding caching strategies on actual Bayesian agent contests")
println()

# Set random seed for reproducibility
Random.seed!(12345)

# Test configuration matching homogeneous-Bayesian-agents.jl
const BENCHMARK_CONFIG = (
    n_agents = 5,           # Number of Bayesian agents
    n_rounds = 500,         # Reduced from 1000 for faster benchmarking  
    χ = 0.05,              # Minimum effort bound
    accuracy = :veryrelaxed, # Use fastest integration for fair comparison
    benchmark_samples = 3,  # Number of benchmark samples per test
    n_reps = 3,            # Number of independent contest runs
    tolerance_levels = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3] # Approximate cache tolerances to test
)

cost(x) = x  # Linear cost function from homogeneous example

"""
Generate a Bayesian agent contest with specified caching strategy
"""
function generate_contest_with_caching(caching::Symbol, cache_tolerance::Float64=1e-12)
    agents = [BayesianAgent(cost; χ=BENCHMARK_CONFIG.χ) for _ in 1:BENCHMARK_CONFIG.n_agents]
    initial_efforts = [rand() for _ in 1:BENCHMARK_CONFIG.n_agents]  # Random as in original
    
    return TullockContest(agents, initial_efforts, BENCHMARK_CONFIG.n_rounds; 
                         accuracy=BENCHMARK_CONFIG.accuracy,
                         caching=caching,
                         cache_tolerance=cache_tolerance)
end

"""
Run contest and collect comprehensive metrics
"""
function benchmark_caching_strategy(caching::Symbol, cache_tolerance::Float64=1e-12, rep::Int=1)
    println("    Rep $rep: Testing $caching caching (tolerance: $(cache_tolerance))...")
    
    # Create contest
    contest = generate_contest_with_caching(caching, cache_tolerance)
    
    # Time the execution
    start_time = time()
    final_round = run!(contest)
    end_time = time()
    runtime = end_time - start_time
    
    # Memory benchmark
    benchmark_result = @benchmark run!(contest_setup) samples=BENCHMARK_CONFIG.benchmark_samples evals=1 setup=(
        contest_setup = generate_contest_with_caching($caching, $cache_tolerance)
    )
    
    # Extract final efforts and convergence metrics
    contest_final_efforts = TullockDynamics.final_efforts(contest)
    effort_mean = mean(contest_final_efforts)
    effort_std = std(contest_final_efforts)
    
    return (
        caching = caching,
        cache_tolerance = cache_tolerance,
        rep = rep,
        runtime = runtime,
        benchmark_time = median(benchmark_result.times) / 1e9, # Convert to seconds
        memory = median(benchmark_result.memory),
        allocations = median(benchmark_result.allocs),
        final_round = final_round,
        final_efforts = contest_final_efforts,
        effort_mean = effort_mean,
        effort_std = effort_std,
        converged = final_round <= BENCHMARK_CONFIG.n_rounds  # Did it converge before max rounds?
    )
end

"""
Find optimal approximate tolerance by testing multiple levels
"""
function find_optimal_approximate_tolerance()
    println("  Finding optimal approximate caching tolerance...")
    
    tolerance_results = []
    
    for tolerance in BENCHMARK_CONFIG.tolerance_levels
        result = benchmark_caching_strategy(:approximate, tolerance, 1)
        push!(tolerance_results, result)
        @printf("    Tolerance %.0e: %.3f s runtime\n", tolerance, result.runtime)
    end
    
    # Find tolerance with best (lowest) runtime
    best_result = tolerance_results[argmin([r.runtime for r in tolerance_results])]
    println("  → Optimal tolerance: $(best_result.cache_tolerance) ($(best_result.runtime:.3f) s)")
    
    return best_result.cache_tolerance
end

"""
Main benchmark execution
"""
function main()
    println("Configuration:")
    println("  Agents: $(BENCHMARK_CONFIG.n_agents)")
    println("  Rounds: $(BENCHMARK_CONFIG.n_rounds)")
    println("  χ (min effort): $(BENCHMARK_CONFIG.χ)")
    println("  Integration accuracy: $(BENCHMARK_CONFIG.accuracy)")
    println("  Benchmark samples: $(BENCHMARK_CONFIG.benchmark_samples)")
    println("  Independent runs: $(BENCHMARK_CONFIG.n_reps)")
    println()
    
    # Results storage
    all_results = []
    
    # 1. Test non-cached (baseline)
    println("Testing :none caching (baseline)...")
    println("-" * "="^50)
    
    none_results = []
    for rep in 1:BENCHMARK_CONFIG.n_reps
        result = benchmark_caching_strategy(:none, 1e-12, rep)
        push!(none_results, result)
        push!(all_results, result)
    end
    
    # 2. Test exact caching
    println("\nTesting :exact caching...")
    println("-" * "="^50)
    
    exact_results = []
    for rep in 1:BENCHMARK_CONFIG.n_reps
        result = benchmark_caching_strategy(:exact, 1e-12, rep)
        push!(exact_results, result)
        push!(all_results, result)
    end
    
    # 3. Find optimal approximate tolerance and test
    println("\nTesting :approximate caching...")
    println("-" * "="^50)
    
    optimal_tolerance = find_optimal_approximate_tolerance()
    
    approx_results = []
    for rep in 1:BENCHMARK_CONFIG.n_reps
        result = benchmark_caching_strategy(:approximate, optimal_tolerance, rep)
        push!(approx_results, result)
        push!(all_results, result)
    end
    
    # Organize results by strategy
    results_by_strategy = Dict(
        :none => none_results,
        :exact => exact_results,
        :approximate => approx_results
    )
    
    # Performance comparison table
    println("\n" * "="^80)
    println("PERFORMANCE COMPARISON")
    println("="^80)
    
    # Runtime comparison
    println("\nRuntime Performance:")
    println("┌─────────────────┬──────────────┬──────────────┬─────────────┬──────────────┐")
    println("│   Caching       │ Mean Runtime │   Std Dev    │   Speedup   │   Memory     │")
    println("│   Strategy      │     (s)      │     (s)      │   vs None   │    (MB)      │")
    println("├─────────────────┼──────────────┼──────────────┼─────────────┼──────────────┤")
    
    none_runtime = mean([r.runtime for r in none_results])
    
    for strategy in [:none, :exact, :approximate]
        results = results_by_strategy[strategy]
        
        avg_runtime = mean([r.runtime for r in results])
        std_runtime = std([r.runtime for r in results])
        speedup = none_runtime / avg_runtime
        avg_memory = mean([r.memory for r in results]) / 1e6  # Convert to MB
        
        tolerance_str = strategy == :approximate ? " (tol=$(optimal_tolerance))" : ""
        
        @printf("│ %-14s%s │ %10.3f   │ %10.3f   │ %9.2fx   │ %10.2f   │\n",
                string(strategy), tolerance_str, avg_runtime, std_runtime, speedup, avg_memory)
    end
    
    println("└─────────────────┴──────────────┴──────────────┴─────────────┴──────────────┘")
    
    # Convergence analysis
    println("\nConvergence Analysis:")
    println("┌─────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    println("│   Caching       │ Mean Effort │  Std Effort │   Converge  │ Mean Final  │")
    println("│   Strategy      │             │             │   Rate      │   Round     │")
    println("├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    
    for strategy in [:none, :exact, :approximate]
        results = results_by_strategy[strategy]
        
        all_efforts = vcat([r.final_efforts for r in results]...)
        mean_effort = mean(all_efforts)
        std_effort = std(all_efforts)
        converge_rate = sum([r.converged for r in results]) / length(results) * 100
        mean_final_round = mean([r.final_round for r in results])
        
        @printf("│ %-15s │ %9.6f   │ %9.6f   │ %9.1f%%  │ %9.1f   │\n",
                string(strategy), mean_effort, std_effort, converge_rate, mean_final_round)
    end
    
    println("└─────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
    
    # Detailed tolerance analysis for approximate caching
    println("\nApproximate Caching Tolerance Analysis:")
    println("┌──────────────┬──────────────┬──────────────┬──────────────┐")
    println("│  Tolerance   │   Runtime    │   Speedup    │   Relative   │")
    println("│              │     (s)      │   vs None    │   to Exact   │")
    println("├──────────────┼──────────────┼──────────────┼──────────────┤")
    
    exact_runtime = mean([r.runtime for r in exact_results])
    
    # Re-test all tolerances for detailed analysis
    for tolerance in BENCHMARK_CONFIG.tolerance_levels
        result = benchmark_caching_strategy(:approximate, tolerance, 1)
        speedup_vs_none = none_runtime / result.runtime
        speedup_vs_exact = exact_runtime / result.runtime
        
        marker = tolerance == optimal_tolerance ? " ←" : ""
        
        @printf("│ %10.0e   │ %10.3f   │ %10.2fx   │ %10.2fx   │%s\n",
                tolerance, result.runtime, speedup_vs_none, speedup_vs_exact, marker)
    end
    
    println("└──────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Summary and recommendations
    println("\n" * "="^80)
    println("SUMMARY AND RECOMMENDATIONS")
    println("="^80)
    
    exact_speedup = none_runtime / mean([r.runtime for r in exact_results])
    approx_speedup = none_runtime / mean([r.runtime for r in approx_results])
    
    println("\nKey Findings:")
    @printf("• No caching (baseline): %.3f s average runtime\n", none_runtime)
    @printf("• Exact caching: %.2fx speedup vs baseline\n", exact_speedup)
    @printf("• Approximate caching: %.2fx speedup vs baseline (optimal tolerance: %.0e)\n", 
            approx_speedup, optimal_tolerance)
    
    # Check for statistical significance
    none_efforts = vcat([r.final_efforts for r in none_results]...)
    exact_efforts = vcat([r.final_efforts for r in exact_results]...)
    approx_efforts = vcat([r.final_efforts for r in approx_results]...)
    
    exact_diff = abs(mean(exact_efforts) - mean(none_efforts)) / mean(none_efforts) * 100
    approx_diff = abs(mean(approx_efforts) - mean(none_efforts)) / mean(none_efforts) * 100
    
    println("\nAccuracy Impact:")
    @printf("• Exact caching: %.4f%% difference in final efforts vs baseline\n", exact_diff)
    @printf("• Approximate caching: %.4f%% difference in final efforts vs baseline\n", approx_diff)
    
    println("\nRecommendations:")
    if exact_speedup >= approx_speedup * 0.95  # Within 5%
        println("✓ Use exact caching - simpler implementation with similar performance")
    elseif approx_speedup > exact_speedup * 1.1  # >10% better
        println("✓ Use approximate caching with tolerance $(optimal_tolerance)")
    else
        println("• Both caching strategies provide similar benefits")
    end
    
    if max(exact_speedup, approx_speedup) > 1.2
        println("✓ Caching provides significant performance improvement for Bayesian agents")
    elseif max(exact_speedup, approx_speedup) > 1.05
        println("• Caching provides modest performance improvement")
    else
        println("⚠ Caching overhead may not justify benefits for this configuration")
    end
    
    # Implementation guidance
    println("\nImplementation:")
    println("Caching is now integrated into TullockContest constructor:")
    println("```julia")
    println("# No caching (default)")
    println("contest = TullockContest(agents, initial_efforts, rounds)")
    println()
    println("# Exact caching")
    println("contest = TullockContest(agents, initial_efforts, rounds; caching=:exact)")
    println()
    println("# Approximate caching")
    @printf("contest = TullockContest(agents, initial_efforts, rounds; caching=:approximate, cache_tolerance=%.0e)\n", optimal_tolerance)
    println("```")
    
    println("\nBenchmark completed successfully!")
    return results_by_strategy
end

# Execute the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end