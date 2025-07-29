#!/usr/bin/env julia
"""
Bayesian Agent Accuracy Benchmark

Compares the performance and convergence of Bayesian agents across three accuracy levels:
- :default (atol=1e-10, reltol=1e-8) - Highest precision
- :relaxed (atol=1e-6, reltol=1e-4) - Balanced precision/speed
- :veryrelaxed (atol=1e-5, reltol=1e-3) - Lower precision, fastest

Metrics compared:
1. Runtime performance
2. Memory allocation
3. Final effort convergence
4. Numerical accuracy of results

Author: TullockDynamics.jl Performance Analysis
"""

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf
using Random
using LinearAlgebra

println("="^80)
println("BAYESIAN AGENTS ACCURACY BENCHMARK")
println("="^80)
println("Comparing integration tolerance levels for Bayesian agent performance")
println()

# Set random seed for reproducibility
Random.seed!(12345)

# Benchmark configuration
const BENCHMARK_CONFIG = (
    n_agents = 5,           # Number of Bayesian agents
    n_rounds = 2000,         # Number of contest rounds
    χ = 0.05,              # Minimum effort bound
    n_samples = 3,          # Number of benchmark samples
    n_reps = 3             # Number of independent contest runs
)

cost(x) = x  # Linear cost function

"""
Generate a Bayesian agent contest with specified accuracy
"""
function generate_contest(accuracy::Symbol)
    agents = [BayesianAgent(cost; χ=BENCHMARK_CONFIG.χ) for _ in 1:BENCHMARK_CONFIG.n_agents]
    initial_efforts = [0.1 + 0.1*rand() for _ in 1:BENCHMARK_CONFIG.n_agents]  # Consistent initial range
    return TullockContest(agents, initial_efforts, BENCHMARK_CONFIG.n_rounds; accuracy=accuracy)
end

"""
Run a single contest and collect detailed metrics
"""
function run_contest_with_metrics(accuracy::Symbol, rep::Int)
    println("    Rep $rep: Creating contest...")
    contest = generate_contest(accuracy)
    
    # Verify tolerance settings
    atol, reltol = contest.workspace.atol, contest.workspace.reltol
    
    println("    Rep $rep: Running contest (tolerances: atol=$atol, reltol=$reltol)...")
    
    # Time the execution
    start_time = time()
    final_round = run!(contest)
    end_time = time()
    runtime = end_time - start_time
    
    # Extract final efforts (last round)
    final_efforts = contest.efforts[:, end]
    
    # Calculate convergence metrics
    effort_mean = mean(final_efforts)
    effort_std = std(final_efforts)
    effort_range = maximum(final_efforts) - minimum(final_efforts)
    
    # Total utility in final round
    final_utilities = contest.utilities[:, end]
    total_utility = sum(final_utilities)
    
    # Nash gap in final round (measure of equilibrium)
    final_nash_gaps = contest.nash_gaps[:, end]
    total_nash_gap = sum(final_nash_gaps)
    
    return (
        accuracy = accuracy,
        rep = rep,
        runtime = runtime,
        final_round = final_round,
        final_efforts = final_efforts,
        effort_mean = effort_mean,
        effort_std = effort_std,
        effort_range = effort_range,
        total_utility = total_utility,
        total_nash_gap = total_nash_gap,
        atol = atol,
        reltol = reltol
    )
end

"""
Run memory benchmark for a specific accuracy level
"""
function benchmark_memory(accuracy::Symbol)
    println("    Running memory benchmark...")
    
    # Create a fresh contest for memory benchmarking
    contest = generate_contest(accuracy)
    
    # Benchmark memory usage
    benchmark_result = @benchmark run!($contest) samples=BENCHMARK_CONFIG.n_samples evals=1 setup=(
        contest = generate_contest($accuracy)
    )
    
    return (
        time = median(benchmark_result).time,
        memory = median(benchmark_result).memory,
        allocs = median(benchmark_result).allocs,
        accuracy = accuracy
    )
end

"""
Compare Nash gaps and convergence quality across accuracy levels
"""
function analyze_nash_gaps_and_convergence(results_by_accuracy)
    println("\n" * "="^80)
    println("NASH GAP AND CONVERGENCE ANALYSIS")
    println("="^80)
    println()
    
    # First, create a focused Nash gap comparison table
    println("OVERALL NASH GAP COMPARISON:")
    println("┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
    println("│   Accuracy      │  Mean Nash Gap  │   Std Nash Gap  │ Nash Gap Range  │")
    println("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
    
    nash_gap_data = Dict()
    
    for accuracy in [:default, :relaxed, :veryrelaxed]
        if haskey(results_by_accuracy, accuracy)
            results = results_by_accuracy[accuracy]
            
            # Collect all Nash gaps from final rounds
            nash_gaps = [r.total_nash_gap for r in results]
            nash_gap_data[accuracy] = nash_gaps
            
            mean_gap = mean(nash_gaps)
            std_gap = std(nash_gaps)
            min_gap = minimum(nash_gaps)
            max_gap = maximum(nash_gaps)
            
            @printf("│ %-15s │ %13.8f   │ %13.8f   │ [%.2e, %.2e] │\n",
                    string(accuracy), mean_gap, std_gap, min_gap, max_gap)
        end
    end
    
    println("└─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
    println()
    
    # Nash gap statistical comparison
    println("NASH GAP STATISTICAL ANALYSIS:")
    
    # Compare each pair of accuracy levels
    accuracies = collect(keys(nash_gap_data))
    if length(accuracies) >= 2
        println()
        for i in 1:length(accuracies)
            for j in (i+1):length(accuracies)
                acc1, acc2 = accuracies[i], accuracies[j]
                gaps1 = nash_gap_data[acc1]
                gaps2 = nash_gap_data[acc2]
                
                # Statistical comparison
                mean_diff = mean(gaps1) - mean(gaps2)
                relative_diff = abs(mean_diff) / mean([mean(gaps1), mean(gaps2)]) * 100
                
                better_accuracy = mean_diff < 0 ? acc1 : acc2
                worse_accuracy = mean_diff < 0 ? acc2 : acc1
                
                @printf("• %s vs %s Nash gaps:\n", acc1, acc2)
                @printf("  Mean difference: %.8f (%.3f%% relative)\n", abs(mean_diff), relative_diff)
                @printf("  %s achieves %.3fx better Nash gap on average\n", 
                        better_accuracy, mean([mean(gaps1), mean(gaps2)]) / min(mean(gaps1), mean(gaps2)))
                println()
            end
        end
    end
    
    println("Detailed convergence metrics by accuracy level:")
    println()
    
    # Group results by accuracy
    for accuracy in [:default, :relaxed, :veryrelaxed]
        if haskey(results_by_accuracy, accuracy)
            results = results_by_accuracy[accuracy]
            
            # Collect all final efforts across reps
            all_efforts = vcat([r.final_efforts for r in results]...)
            
            println("$accuracy accuracy:")
            @printf("  Mean effort: %.6f ± %.6f\n", mean(all_efforts), std(all_efforts))
            @printf("  Range: [%.6f, %.6f]\n", minimum(all_efforts), maximum(all_efforts))
            @printf("  Final round reached: %d ± %.1f\n", 
                    round(Int, mean([r.final_round for r in results])),
                    std([r.final_round for r in results]))
            
            # Nash gap (measure of equilibrium quality)
            avg_nash_gap = mean([r.total_nash_gap for r in results])
            @printf("  Avg Nash gap: %.8f\n", avg_nash_gap)
            
            # Total utility (measure of efficiency)
            avg_utility = mean([r.total_utility for r in results])
            @printf("  Avg total utility: %.6f\n", avg_utility)
            println()
        end
    end
    
    # Statistical comparison between accuracy levels
    if length(results_by_accuracy) >= 2
        println("Statistical Differences:")
        accuracies = collect(keys(results_by_accuracy))
        
        for i in 1:length(accuracies)
            for j in (i+1):length(accuracies)
                acc1, acc2 = accuracies[i], accuracies[j]
                efforts1 = vcat([r.final_efforts for r in results_by_accuracy[acc1]]...)
                efforts2 = vcat([r.final_efforts for r in results_by_accuracy[acc2]]...)
                
                # Simple statistical test (difference in means)
                mean_diff = abs(mean(efforts1) - mean(efforts2))
                pooled_std = sqrt((var(efforts1) + var(efforts2)) / 2)
                relative_diff = mean_diff / mean([mean(efforts1), mean(efforts2)]) * 100
                
                @printf("  %s vs %s: Mean difference = %.6f (%.3f%% relative)\n", 
                        acc1, acc2, mean_diff, relative_diff)
            end
        end
    end
end

"""
Main benchmark execution
"""
function main()
    println("Configuration:")
    println("  Agents: $(BENCHMARK_CONFIG.n_agents)")
    println("  Rounds: $(BENCHMARK_CONFIG.n_rounds)")
    println("  χ (min effort): $(BENCHMARK_CONFIG.χ)")
    println("  Benchmark samples: $(BENCHMARK_CONFIG.n_samples)")
    println("  Independent runs: $(BENCHMARK_CONFIG.n_reps)")
    println()
    
    accuracies = [:default, :relaxed, :veryrelaxed]
    results_by_accuracy = Dict()
    memory_results = Dict()
    
    for accuracy in accuracies
        println("Testing $accuracy accuracy...")
        println("-" * "="^50)
        
        # Run multiple independent contests
        contest_results = []
        for rep in 1:BENCHMARK_CONFIG.n_reps
            result = run_contest_with_metrics(accuracy, rep)
            push!(contest_results, result)
        end
        results_by_accuracy[accuracy] = contest_results
        
        # Memory benchmark
        memory_result = benchmark_memory(accuracy)
        memory_results[accuracy] = memory_result
        
        # Summary for this accuracy level
        runtimes = [r.runtime for r in contest_results]
        println("\n  Runtime Summary:")
        @printf("    Mean: %.3f ± %.3f seconds\n", mean(runtimes), std(runtimes))
        @printf("    Range: [%.3f, %.3f] seconds\n", minimum(runtimes), maximum(runtimes))
        
        println("  Memory Summary:")
        @printf("    Time: %.2f ms\n", memory_result.time / 1e6)
        @printf("    Memory: %s\n", BenchmarkTools.prettymemory(memory_result.memory))
        @printf("    Allocations: %d\n", memory_result.allocs)
        @printf("    Tolerances: atol=%.0e, reltol=%.0e\n", 
                contest_results[1].atol, contest_results[1].reltol)
        
        println("\n" * "✓" * " Completed $accuracy accuracy\n")
    end
    
    # Performance comparison table
    println("="^80)
    println("PERFORMANCE COMPARISON")
    println("="^80)
    
    println("\nRuntime Performance:")
    println("┌─────────────┬──────────────┬──────────────┬─────────────┬──────────────┐")
    println("│   Accuracy  │ Mean Runtime │   Memory     │ Allocations │   Speedup    │")
    println("├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤")
    
    default_time = mean([r.runtime for r in results_by_accuracy[:default]])
    
    for accuracy in accuracies
        contest_results = results_by_accuracy[accuracy]
        memory_result = memory_results[accuracy]
        
        avg_runtime = mean([r.runtime for r in contest_results])
        speedup = default_time / avg_runtime
        
        @printf("│ %-11s │ %8.3f s   │ %10s   │ %9d   │ %8.2fx    │\n",
                string(accuracy), avg_runtime, 
                BenchmarkTools.prettymemory(memory_result.memory),
                memory_result.allocs, speedup)
    end
    
    println("└─────────────┴──────────────┴──────────────┴─────────────┴──────────────┘")
    
    # Tolerance settings table
    println("\nTolerance Settings:")
    println("┌─────────────┬─────────────┬──────────────┐")
    println("│   Accuracy  │    atol     │    reltol    │")
    println("├─────────────┼─────────────┼──────────────┤")
    
    for accuracy in accuracies
        result = results_by_accuracy[accuracy][1]  # Get tolerances from first result
        @printf("│ %-11s │ %11.0e │ %12.0e │\n", 
                string(accuracy), result.atol, result.reltol)
    end
    
    println("└─────────────┴─────────────┴──────────────┘")
    
    # Nash gap and convergence analysis
    analyze_nash_gaps_and_convergence(results_by_accuracy)
    
    # Final summary
    println("="^80)
    println("SUMMARY")
    println("="^80)
    
    relaxed_speedup = default_time / mean([r.runtime for r in results_by_accuracy[:relaxed]])
    veryrelaxed_speedup = default_time / mean([r.runtime for r in results_by_accuracy[:veryrelaxed]])
    
    # Nash gap analysis for summary
    default_nash_gaps = [r.total_nash_gap for r in results_by_accuracy[:default]]
    relaxed_nash_gaps = [r.total_nash_gap for r in results_by_accuracy[:relaxed]]  
    veryrelaxed_nash_gaps = [r.total_nash_gap for r in results_by_accuracy[:veryrelaxed]]
    
    println("Key Findings:")
    @printf("• relaxed accuracy: %.2fx speedup vs default\n", relaxed_speedup)
    @printf("• veryrelaxed accuracy: %.2fx speedup vs default\n", veryrelaxed_speedup)
    
    println("\nNash Gap Quality (lower is better):")
    @printf("• default accuracy: %.8f mean Nash gap\n", mean(default_nash_gaps))
    @printf("• relaxed accuracy: %.8f mean Nash gap\n", mean(relaxed_nash_gaps))
    @printf("• veryrelaxed accuracy: %.8f mean Nash gap\n", mean(veryrelaxed_nash_gaps))
    
    # Nash gap degradation analysis
    relaxed_nash_degradation = (mean(relaxed_nash_gaps) - mean(default_nash_gaps)) / mean(default_nash_gaps) * 100
    veryrelaxed_nash_degradation = (mean(veryrelaxed_nash_gaps) - mean(default_nash_gaps)) / mean(default_nash_gaps) * 100
    
    @printf("• relaxed Nash gap degradation: %.3f%% vs default\n", relaxed_nash_degradation)
    @printf("• veryrelaxed Nash gap degradation: %.3f%% vs default\n", veryrelaxed_nash_degradation)
    
    # Check if speedups match expectations
    expected_relaxed = 2.3
    expected_veryrelaxed = 6.9
    
    if relaxed_speedup < expected_relaxed * 0.8
        println("⚠️  relaxed speedup lower than expected (~2.3x)")
    end
    
    if veryrelaxed_speedup < expected_veryrelaxed * 0.5
        println("⚠️  veryrelaxed speedup much lower than expected (~6.9x)")
    end
    
    println("\nRecommendations (considering both speed and Nash gap quality):")
    
    # Quality thresholds for Nash gap degradation
    acceptable_degradation = 5.0  # 5% degradation acceptable
    significant_degradation = 20.0  # 20% degradation significant
    
    if relaxed_speedup > 1.5 && abs(relaxed_nash_degradation) < acceptable_degradation
        println("✓ Use :relaxed for balanced performance/accuracy - minimal Nash gap impact")
    elseif relaxed_speedup > 1.5 && abs(relaxed_nash_degradation) < significant_degradation
        println("• Use :relaxed for good performance/accuracy tradeoff - small Nash gap impact")
    elseif relaxed_speedup > 1.5
        println("⚠ :relaxed provides speedup but with notable Nash gap degradation")
    end
    
    if veryrelaxed_speedup > 2.0 && abs(veryrelaxed_nash_degradation) < acceptable_degradation
        println("✓ Use :veryrelaxed for maximum speed - minimal Nash gap impact")
    elseif veryrelaxed_speedup > 2.0 && abs(veryrelaxed_nash_degradation) < significant_degradation
        println("• Use :veryrelaxed for maximum speed when small accuracy loss is acceptable")
    elseif veryrelaxed_speedup > 2.0
        println("⚠ :veryrelaxed provides significant speedup but with notable Nash gap degradation")
    end
    
    # Overall recommendation based on combined metrics
    if abs(relaxed_nash_degradation) < acceptable_degradation && relaxed_speedup > 1.5
        println("\n🎯 OVERALL RECOMMENDATION: :relaxed accuracy provides the best speed/quality balance")
    elseif abs(veryrelaxed_nash_degradation) < acceptable_degradation && veryrelaxed_speedup > 2.0
        println("\n🎯 OVERALL RECOMMENDATION: :veryrelaxed accuracy provides excellent speedup with minimal quality loss")
    else
        println("\n🎯 OVERALL RECOMMENDATION: :default accuracy for applications requiring highest Nash equilibrium quality")
    end
    
    # Accuracy impact
    default_results = results_by_accuracy[:default]
    relaxed_results = results_by_accuracy[:relaxed]
    veryrelaxed_results = results_by_accuracy[:veryrelaxed]
    
    default_efforts = vcat([r.final_efforts for r in default_results]...)
    relaxed_efforts = vcat([r.final_efforts for r in relaxed_results]...)
    veryrelaxed_efforts = vcat([r.final_efforts for r in veryrelaxed_results]...)
    
    relaxed_error = abs(mean(relaxed_efforts) - mean(default_efforts)) / mean(default_efforts) * 100
    veryrelaxed_error = abs(mean(veryrelaxed_efforts) - mean(default_efforts)) / mean(default_efforts) * 100
    
    @printf("\nAccuracy Impact on Final Efforts:\n")
    @printf("• relaxed vs default: %.3f%% difference\n", relaxed_error)
    @printf("• veryrelaxed vs default: %.3f%% difference\n", veryrelaxed_error)
    
    if relaxed_error < 1.0
        println("✓ relaxed accuracy maintains high precision")
    end
    if veryrelaxed_error < 5.0
        println("✓ veryrelaxed accuracy acceptable for most applications")
    end
    
    println("\nNASH GAP INTERPRETATION:")
    println("Nash gap represents the distance from Nash equilibrium - lower values indicate")
    println("better convergence to the theoretical optimal state. The analysis above shows")
    println("how integration accuracy affects equilibrium quality in Bayesian agent contests.")
    println()
    println("A Nash gap degradation of:")
    println("• < 5%: Negligible impact on equilibrium quality")
    println("• 5-20%: Small but measurable impact, often acceptable for performance gains")
    println("• > 20%: Significant impact, may affect economic interpretation of results")
    
    println("\nBenchmark completed successfully!")
end

# Execute the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end