#!/usr/bin/env julia

"""
Quick benchmark comparing cached vs non-cached implementations
"""

using Pkg
Pkg.activate(".")

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf

println("🚀 Quick TullockDynamics.jl Caching Benchmark")
println("Current branch: ", read(`git branch --show-current`, String) |> strip)
println("="^60)

# Define test configurations
linear_cost(x) = x
quadratic_cost(x) = 0.5 * x^2

configs = [
    ("Small Contest", [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(linear_cost)], [0.1, 0.15, 0.2], 30),
    ("Medium Contest", [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(linear_cost), StandardAgent(quadratic_cost)], [0.1, 0.15, 0.18, 0.22], 50),
    ("Large Contest", [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(linear_cost), StandardAgent(quadratic_cost), MLEAgent(quadratic_cost)], [0.1, 0.12, 0.15, 0.18, 0.2], 75)
]

results = []

for (name, agents, efforts, rounds) in configs
    println("\n📊 Testing: $name ($rounds rounds, $(length(agents)) agents)")
    
    # Clear caches
    clear_agent_caches!()
    clear_bayesian_cache!()
    GC.gc()
    
    # Benchmark
    result = @benchmark begin
        contest = TullockContest($agents, $efforts, $rounds)
        final_round = run!(contest)
        sum(contest.efforts[:, min(final_round, $rounds)])
    end samples=3 seconds=15
    
    # Cache stats
    est_cache = length(TullockDynamics.ESTIMATOR_CACHE)
    br_cache = length(TullockDynamics.BEST_RESPONSE_CACHE)
    
    @printf "  Time:   %6.2f ms\n" median(result.times) / 1e6
    @printf "  Memory: %6.1f MB\n" result.memory / 1e6
    @printf "  Caches: %4d estimator + %4d best_response\n" est_cache br_cache
    
    push!(results, (name=name, time=median(result.times)/1e6, memory=result.memory/1e6, est_cache=est_cache, br_cache=br_cache))
end

println("\n" * "="^60)
println("CACHED IMPLEMENTATION SUMMARY")
println("="^60)
for result in results
    @printf "%-15s: %6.2f ms | %5.1f MB | %3d+%3d cache\n" result.name result.time result.memory result.est_cache result.br_cache
end

println("\n✅ Cached benchmark completed!")
println("\n💡 Next steps:")
println("   1. Switch to 'improved' branch: git checkout improved")  
println("   2. Run: julia quick_benchmark.jl")
println("   3. Compare the timing results")