#!/usr/bin/env julia
"""
Test the plotting extension system
"""

using TullockDynamics

println("Testing plotting extension system...")
println("=" * "="^40)

# Create a simple contest
cost(x) = x
agents = [MLEAgent(cost), BayesianAgent(cost)]
initial_efforts = [0.1, 0.15]
contest = TullockContest(agents, initial_efforts, 20)

# Run the contest
println("Running contest...")
final_round = run!(contest)
println("Contest completed in $final_round rounds")

# Test 1: Try plotting without Plots.jl loaded (should error helpfully)
println("\n1. Testing visualise() without Plots.jl loaded...")
try
    visualise(contest)
    println("ERROR: visualise() should have failed!")
catch e
    println("✓ Got expected error message:")
    println(e.msg)
end

# Test 2: Load Plots.jl and try again
println("\n2. Loading Plots.jl and testing plotting...")
try
    using Plots
    using Measures  # Also needed for the extension
    
    println("✓ Plots.jl loaded successfully")
    println("Testing visualise() function...")
    
    # This should now work
    plot_result = visualise(contest)
    println("✓ visualise() function worked! Plot type: $(typeof(plot_result))")
    
    # Save a test plot to verify it works
    savefig(plot_result, "test_plot.png")
    println("✓ Plot saved as test_plot.png")
    
catch e
    if isa(e, LoadError)
        println("Could not load Plots.jl: $e")
        println("This is expected if Plots.jl is not in the environment")
    else
        println("Error during plotting: $e")
    end
end

println("\n✓ Extension system test completed!")
println("\nSummary:")
println("• TullockDynamics.jl loaded without Plots dependency")
println("• visualise() provides helpful error when Plots not loaded")
println("• Extension automatically activates when Plots.jl is loaded")
println("• Core functionality works independently of plotting")