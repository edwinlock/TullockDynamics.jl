using Test
using TullockDynamics

@testset "Plotting Extension Tests" begin
    
    # Create a test contest
    cost(x) = x
    agents = [MLEAgent(cost; χ=0.05), BayesianAgent(cost; χ=0.05)]
    initial_efforts = [0.1, 0.15]
    contest = TullockContest(agents, initial_efforts, 10)
    run!(contest)
    
    @testset "Extension Loading Behavior" begin
        
        @testset "visualise stub function exists" begin
            # Test that the stub function is available
            @test hasmethod(visualise, (TullockContest,))
            
            # Test that it provides helpful error without Plots
            @test_throws ErrorException visualise(contest)
            
            # Test error message content
            try
                visualise(contest)
                @test false  # Should not reach here
            catch e
                @test occursin("Plots.jl", e.msg)
                @test occursin("using Plots", e.msg)
            end
        end
        
        @testset "Core functionality independent of Plots" begin
            # Test that all core functions work without Plots
            @test TullockDynamics.final_efforts(contest) isa Vector{Float64}
            @test TullockDynamics.convergence_status(contest, 10) isa Tuple
            
            # Test contest creation and running
            new_contest = TullockContest(agents, initial_efforts, 5)
            @test run!(new_contest) isa Int
            
            # Test that Nash gap computation works
            @test sum(new_contest.nash_gaps[:, end]) isa Float64
        end
    end
    
    @testset "Extension Activation (conditional)" begin
        # This test only runs if Plots.jl is available
        plots_available = false
        
        try
            # Try to load Plots and Measures
            eval(:(using Plots, Measures))
            plots_available = true
        catch LoadError
            @test_skip "Plots.jl not available - skipping extension tests"
        end
        
        if plots_available
            @testset "Plots Extension Active" begin
                # Test that visualise now works properly
                result = visualise(contest)
                @test result isa Any  # Should return a plot object
                
                # Test that the function accepts keyword arguments
                result_with_kwargs = visualise(contest; rounds=(1, 5))
                @test result_with_kwargs isa Any
                
                # Test with different ylims
                result_ylims = visualise(contest; ylims=(0, 1))
                @test result_ylims isa Any
            end
        end
    end
    
    @testset "Extension Configuration" begin
        # Test that the package loads without plotting dependencies
        @test !("Plots" in [string(name) for name in keys(Base.loaded_modules) if name.name != "TullockDynamics"])
        
        # Test that visualise is exported
        @test :visualise in names(TullockDynamics)
        
        # Test that core exports are still available
        core_exports = [:MLEAgent, :BayesianAgent, :TullockContest, :run!, :final_efforts, :convergence_status]
        for export_name in core_exports
            @test export_name in names(TullockDynamics)
        end
    end
end