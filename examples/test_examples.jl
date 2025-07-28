#!/usr/bin/env julia
"""
Test script to verify all examples run successfully.

Usage: julia test_examples.jl
"""

using TullockDynamics

function test_example(filename::String)
    println("Testing $filename...")
    try
        include(filename)
        println("  ✓ Success")
        return true
    catch e
        println("  ✗ Error: $e")
        return false
    end
end

function main()
    println("Testing TullockDynamics.jl examples...")
    println("=" ^ 50)
    
    examples = [
        "homogeneous-linear-MLE-agents.jl",
        "homogeneous-linear-DetMLE-agents.jl", 
        "homogeneous-linear-dumb-agents.jl",
        "two-linear-mle-agents.jl",
        "homogeneous-Bayesian-agents.jl"
    ]
    
    successes = 0
    for example in examples
        if test_example(example)
            successes += 1
        end
    end
    
    println("=" ^ 50)
    println("Results: $successes/$(length(examples)) examples passed")
    
    if successes == length(examples)
        println("🎉 All examples working correctly!")
        return true
    else
        println("⚠️  Some examples failed - check output above")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end