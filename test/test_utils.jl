@testset "Utils Tests" begin
    
    @testset "find_root function" begin
        # Test simple decreasing function: f(x) = 10 - x, root at x = 10
        f1(x) = 10.0 - x
        root = find_root(f1, 0.0)
        @test abs(f1(root)) < 1e-6
        @test root ≈ 10.0 atol=1e-6
        
        # Test with different starting point
        root2 = find_root(f1, 5.0)
        @test abs(f1(root2)) < 1e-6
        @test root2 ≈ 10.0 atol=1e-6
        
        # Test exponential decay: f(x) = e^(-x) - 0.1, root ≈ 2.3
        f2(x) = exp(-x) - 0.1
        root = find_root(f2, 0.0)
        @test abs(f2(root)) < 1e-6  # Relaxed tolerance for numerical precision
        @test root ≈ -log(0.1) atol=1e-5
        
        # Test edge case: function always zero
        f3(x) = 0.0
        root = find_root(f3, 1.0)
        @test root >= 1.0
        @test abs(f3(root)) < 1e-10
        
        # Test precondition: f(l) >= 0
        f4(x) = x - 10.0  # f(0) = -10 < 0, should fail
        @test_throws AssertionError find_root(f4, 0.0)
    end
    
    @testset "max_agent_effort function" begin
        # Linear cost: c(x) = x, max effort when c(x) = 1, so x = 1
        cost1(x) = x
        max_eff = max_agent_effort(cost1)
        @test cost1(max_eff) ≈ 1.0 atol=1e-6
        @test max_eff ≈ 1.0 atol=1e-6
        
        # Quadratic cost: c(x) = 0.5x², max effort when c(x) = 1, so x = √2
        cost2(x) = 0.5 * x^2
        max_eff = max_agent_effort(cost2)
        @test cost2(max_eff) ≈ 1.0 atol=1e-6
        @test max_eff ≈ sqrt(2.0) atol=1e-6
        
        # Cubic cost: c(x) = x³, max effort when c(x) = 1, so x = 1
        cost3(x) = x^3
        max_eff = max_agent_effort(cost3)
        @test cost3(max_eff) ≈ 1.0 atol=1e-6
        @test max_eff ≈ 1.0 atol=1e-6
    end
    
    @testset "convergence_status function" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 10)
        
        # Test converged case
        converged, actual_round = convergence_status(contest, 5)
        @test converged == true
        @test actual_round == 5
        
        # Test non-converged case (T+1)
        converged, actual_round = convergence_status(contest, 11)
        @test converged == false
        @test actual_round == 10  # Should return T, not T+1
        
        # Test boundary case (exactly T)
        converged, actual_round = convergence_status(contest, 10)
        @test converged == true
        @test actual_round == 10
    end
    
    @testset "final_efforts function" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 5)
        
        # Set some effort values
        contest.efforts[:, 5] = [0.3, 0.4]
        
        final_eff = final_efforts(contest)
        @test final_eff == [0.3, 0.4]
        @test length(final_eff) == 2
    end
    
    @testset "Base.show method" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 3)
        
        # Test that show doesn't crash
        io = IOBuffer()
        show(io, "text/plain", contest)
        output = String(take!(io))
        
        @test occursin("TullockContest", output)
        @test occursin("Number of agents: 2", output)
        @test occursin("Efforts:", output)
        @test occursin("Winners:", output)
        @test occursin("Utilities:", output)
    end
end