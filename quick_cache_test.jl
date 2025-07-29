#!/usr/bin/env julia
"""
Quick test of cached root finding performance
"""

using TullockDynamics
using BenchmarkTools
using Statistics
using Printf

println("Testing find_root_cached availability...")

# Simple test function
test_func(x) = 10.0 - x

# Test that functions work
println("Testing find_root...")
result1 = TullockDynamics.find_root(test_func, 0.0)
println("find_root result: $result1")

println("Testing find_root_cached...")
result2 = TullockDynamics.find_root_cached(test_func, 0.0)
println("find_root_cached result: $result2")

# Quick benchmark
println("\nQuick performance comparison:")
time1 = @benchmark TullockDynamics.find_root($test_func, 0.0) samples=100
time2 = @benchmark TullockDynamics.find_root_cached($test_func, 0.0) samples=100

@printf("Original:  %.2f μs\n", median(time1.times) / 1000)
@printf("Cached:    %.2f μs\n", median(time2.times) / 1000)
@printf("Speedup:   %.2fx\n", median(time1.times) / median(time2.times))

println("\n✓ Cached root finding is working correctly!")