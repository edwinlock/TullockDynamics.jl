#!/usr/bin/env julia

"""
Fast benchmark comparing branches - minimal samples for quick comparison
"""

using Pkg
Pkg.activate(".")

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf

println("⚡ Fast TullockDynamics.jl Benchmark")
println("Current branch: ", read(`git branch --show-current`, String) |> strip)
println("="^50)

# Single test configuration 
linear_cost(x) = x
quadratic_cost(x) = 0.5 * x^2

agents = [MLEAgent(linear_cost), DetMLEAgent(quadratic_cost), DumbAgent(linear_cost)]
efforts = [0.1, 0.15, 0.2] 
rounds = 40

println("\n📊 Testing: 3 agents, 40 rounds")

# Clear caches if available
try
    clear_bayesian_cache!()
    println("  (Bayesian cache cleared)")
catch
end

try 
    clear_agent_caches!()
    println("  (Agent caches cleared)")
catch
end

GC.gc()

# Single quick benchmark
result = @benchmark begin
    contest = TullockContest($agents, $efforts, $rounds)
    final_round = run!(contest)
    sum(contest.efforts[:, min(final_round, $rounds)])
end samples=2 seconds=10

@printf "\nResults:\n"
@printf "  Time:   %6.2f ms\n" median(result.times) / 1e6
@printf "  Memory: %6.1f MB\n" result.memory / 1e6
@printf "  Allocs: %6d\n" result.allocs

# Cache stats if available
try
    est_cache = length(TullockDynamics.ESTIMATOR_CACHE)
    br_cache = length(TullockDynamics.BEST_RESPONSE_CACHE)
    @printf "  Caches: %4d estimator + %4d best_response\n" est_cache br_cache
catch
    println("  Caches: Not available (non-cached branch)")
end

println("\n✅ Fast benchmark completed!")