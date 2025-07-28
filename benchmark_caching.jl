#!/usr/bin/env julia

"""
Comprehensive benchmark comparing cached vs non-cached TullockDynamics implementations.

This script compares the performance of:
1. 'improved' branch: Non-cached implementation with workspace optimizations
2. 'improved-caching' branch: Cached implementation with comprehensive caching

Tests various contest configurations and agent types to measure performance improvements.
"""

using Pkg
Pkg.activate(".")

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf

# Ensure we're on the correct branch
println("Current git branch: ", read(`git branch --show-current`, String) |> strip)

"""
Benchmark configuration structure
"""
struct BenchmarkConfig
    name::String
    agents::Vector{Agent}
    initial_efforts::Vector{Float64}
    num_rounds::Int
    description::String
end

"""
Create various benchmark configurations for testing
"""
function create_benchmark_configs()
    # Define common cost functions
    linear_cost(x) = x
    quadratic_cost(x) = 0.5 * x^2
    cubic_cost(x) = 0.33 * x^3
    
    configs = BenchmarkConfig[]
    
    # Small contests - 3 agents, 30 rounds
    push!(configs, BenchmarkConfig(
        "Small Mixed Contest",
        [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(cubic_cost)],
        [0.1, 0.15, 0.2],
        30,
        "3 agents, 30 rounds, mixed learning algorithms"
    ))
    
    # Medium contests - 5 agents, 50 rounds  
    push!(configs, BenchmarkConfig(
        "Medium MLE Contest",
        [MLEAgent(linear_cost), MLEAgent(quadratic_cost), MLEAgent(cubic_cost), 
         DetMLEAgent(linear_cost), DetMLEAgent(quadratic_cost)],
        [0.1, 0.12, 0.15, 0.18, 0.2],
        50,
        "5 agents, 50 rounds, MLE variants"
    ))
    
    # Large contests - 8 agents, 75 rounds
    push!(configs, BenchmarkConfig(
        "Large Diverse Contest", 
        [MLEAgent(linear_cost), MLEAgent(quadratic_cost), 
         DetMLEAgent(linear_cost), DetMLEAgent(quadratic_cost),
         DumbAgent(linear_cost), DumbAgent(quadratic_cost),
         StandardAgent(linear_cost), StandardAgent(quadratic_cost)],
        [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24],
        75,
        "8 agents, 75 rounds, all algorithm types except Bayesian"
    ))
    
    # Computational intensive - Bayesian agents
    push!(configs, BenchmarkConfig(
        "Bayesian Contest",
        [BayesianAgent(linear_cost), BayesianAgent(quadratic_cost), MLEAgent(linear_cost)],
        [0.1, 0.15, 0.2],
        25,  # Shorter due to computational complexity
        "3 agents including Bayesian, 25 rounds"
    ))
    
    # Memory stress test - long contest
    push!(configs, BenchmarkConfig(
        "Long Contest",
        [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(linear_cost), StandardAgent(quadratic_cost)],
        [0.1, 0.15, 0.18, 0.22],
        100,
        "4 agents, 100 rounds, memory stress test"
    ))
    
    return configs
end

"""
Run a single benchmark configuration and return timing results
"""
function benchmark_configuration(config::BenchmarkConfig; num_samples::Int=5)
    println("\n" * "="^60)
    println("Benchmarking: $(config.name)")
    println("Description: $(config.description)")
    println("="^60)
    
    # Clear all caches before each benchmark
    clear_agent_caches!()
    clear_bayesian_cache!()
    GC.gc()  # Force garbage collection
    
    # Benchmark the contest simulation
    benchmark_result = @benchmark begin
        contest = TullockContest($(config.agents), $(config.initial_efforts), $(config.num_rounds))
        final_round = run!(contest)
        # Return some basic info to prevent optimization
        (final_round, sum(contest.efforts[:, min(final_round, $(config.num_rounds))]))
    end samples=num_samples seconds=30
    
    # Get cache statistics
    estimator_cache_size = length(TullockDynamics.ESTIMATOR_CACHE)
    best_response_cache_size = length(TullockDynamics.BEST_RESPONSE_CACHE) 
    bayesian_cache_size = length(TullockDynamics.BAYESIAN_INTEGRATION_CACHE)
    
    return (
        benchmark = benchmark_result,
        estimator_cache_size = estimator_cache_size,
        best_response_cache_size = best_response_cache_size,
        bayesian_cache_size = bayesian_cache_size
    )
end

"""
Format and display benchmark results
"""
function display_results(config_name::String, result)
    bench = result.benchmark
    
    println("\nResults for: $config_name")
    println("-"^40)
    @printf "  Mean time:     %8.2f ms\n" mean(bench.times) / 1e6
    @printf "  Median time:   %8.2f ms\n" median(bench.times) / 1e6  
    @printf "  Min time:      %8.2f ms\n" minimum(bench.times) / 1e6
    @printf "  Max time:      %8.2f ms\n" maximum(bench.times) / 1e6
    @printf "  Memory alloc:  %8.2f MB\n" bench.memory / 1e6
    @printf "  Allocations:   %8d\n" bench.allocs
    
    println("\nCache Statistics:")
    @printf "  Estimator cache:     %6d entries\n" result.estimator_cache_size
    @printf "  Best response cache: %6d entries\n" result.best_response_cache_size  
    @printf "  Bayesian cache:      %6d entries\n" result.bayesian_cache_size
end

"""
Compare two benchmark results and compute improvement metrics
"""
function compare_results(baseline_name::String, baseline_result, 
                        improved_name::String, improved_result)
    baseline_time = median(baseline_result.benchmark.times)
    improved_time = median(improved_result.benchmark.times)
    
    baseline_memory = baseline_result.benchmark.memory
    improved_memory = improved_result.benchmark.memory
    
    time_speedup = baseline_time / improved_time
    memory_reduction = (baseline_memory - improved_memory) / baseline_memory * 100
    
    println("\n" * "="^60)
    println("PERFORMANCE COMPARISON")
    println("="^60)
    println("$baseline_name vs $improved_name")
    println("-"^60)
    
    @printf "Time improvement:     %.2fx speedup\n" time_speedup
    @printf "Memory improvement:   %.1f%% reduction\n" memory_reduction
    
    if time_speedup > 1.1
        println("✅ Significant performance improvement!")
    elseif time_speedup > 1.0
        println("🟡 Modest performance improvement")  
    else
        println("❌ Performance regression")
    end
    
    return (time_speedup=time_speedup, memory_reduction=memory_reduction)
end

"""
Main benchmark function
"""
function run_comprehensive_benchmark()
    println("🚀 TullockDynamics.jl Caching Performance Benchmark")
    println("=" * "="^60)
    println("Comparing cached vs non-cached implementations")
    println("Current branch: improved-caching (cached implementation)")
    println()
    
    configs = create_benchmark_configs()
    
    # Store results for summary
    all_results = []
    
    for config in configs
        try
            println("\n🔄 Testing configuration: $(config.name)")
            result = benchmark_configuration(config, num_samples=3)
            display_results(config.name, result)
            
            push!(all_results, (config=config, benchmark_result=result))
            
        catch e
            println("❌ Error in $(config.name): $e")
            continue
        end
        
        # Small delay between benchmarks
        sleep(1)
    end
    
    # Summary statistics
    println("\n" * "="^80)
    println("BENCHMARK SUMMARY - CACHED IMPLEMENTATION")
    println("="^80)
    
    for (config, benchmark_result) in all_results
        bench = benchmark_result.benchmark
        @printf "%-25s: %7.2f ms | %6.1f MB | %6d + %4d + %4d cache entries\n" config.name[1:min(24,end)] mean(bench.times)/1e6 bench.memory/1e6 benchmark_result.estimator_cache_size benchmark_result.best_response_cache_size benchmark_result.bayesian_cache_size
    end
    
    println("\nCache Legend: Estimator + Best Response + Bayesian")
    println("\n✅ Benchmark completed successfully!")
    println("💡 To compare with non-cached version:")
    println("   1. Switch to 'improved' branch: git checkout improved") 
    println("   2. Run this same benchmark")
    println("   3. Compare the timing results")
    
    return all_results
end

# Run the benchmark if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_comprehensive_benchmark()
end