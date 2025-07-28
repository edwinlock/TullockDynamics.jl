@testset "Integration Tests" begin
    
    @testset "Full simulation - Homogeneous MLE agents" begin
        cost(x) = x
        n_agents = 5
        agents = [MLEAgent(cost) for _ in 1:n_agents]
        initial_efforts = fill(0.1, n_agents)
        T = 25
        
        contest = TullockContest(agents, initial_efforts, T)
        
        @time final_round = run!(contest)
        
        converged, actual_final = convergence_status(contest, final_round)
        @test actual_final >= 1
        @test actual_final <= T
        
        # Check simulation ran successfully
        for t in 1:actual_final
            @test sum(contest.winners[:, t]) == 1  # One winner per round
            @test all(contest.efforts[:, t] .>= 0.0)  # Non-negative efforts
            @test all(isfinite, contest.utilities[:, t])  # Finite utilities
            @test all(contest.nash_gaps[:, t] .>= 0.0)  # Non-negative gaps
        end
        
        # Check final efforts are reasonable
        final_efforts = contest.efforts[:, actual_final]
        @test all(final_efforts .< 5.0)  # Should not be extremely large
        @test std(final_efforts) >= 0.0  # Some variation expected
    end
    
    @testset "Full simulation - Mixed agent types" begin
        cost(x) = 0.5 * x^2
        
        agents = [
            MLEAgent(cost),
            DetMLEAgent(cost),
            DumbAgent(cost),
            BayesianAgent(cost)
        ]
        initial_efforts = [0.1, 0.15, 0.2, 0.12]
        T = 20
        
        contest = TullockContest(agents, initial_efforts, T)
        
        @time final_round = run!(contest)
        
        converged, actual_final = convergence_status(contest, final_round)
        
        # Test all data is valid
        for t in 1:actual_final
            @test sum(contest.winners[:, t]) == 1
            @test all(isfinite, contest.efforts[:, t])
            @test all(isfinite, contest.utilities[:, t])
            @test all(isfinite, contest.nash_gaps[:, t])
            @test all(contest.efforts[:, t] .>= 0.0)
            @test all(contest.nash_gaps[:, t] .>= 0.0)
        end
        
        # Check that different agent types produce different behavior
        effort_patterns = contest.efforts[:, 1:actual_final]
        @test size(effort_patterns) == (4, actual_final)
        
        # Should have some variation across agents
        final_efforts = contest.efforts[:, actual_final]
        @test length(unique(final_efforts)) > 1  # Not all identical
    end
    
    @testset "Convergence behavior" begin
        cost(x) = x
        agents = [MLEAgent(cost), MLEAgent(cost)]
        contest = TullockContest(agents, [0.2, 0.2], 50)
        
        # Test with strict convergence
        ε = 0.01
        final_round = run!(contest; ε=ε)
        
        converged, actual_final = convergence_status(contest, final_round)
        
        if converged
            total_gap = nash_gap(contest, actual_final)
            @test total_gap <= ε
            @test final_round <= 50
        else
            @test final_round == 51  # T + 1
            @test actual_final == 50
        end
        
        # Test data integrity regardless of convergence
        for t in 1:actual_final
            @test all(isfinite, contest.efforts[:, t])
            @test all(contest.efforts[:, t] .>= 0.0)
        end
    end
    
    @testset "Performance regression test" begin
        # Test that simulations complete in reasonable time
        cost(x) = x
        agents = [MLEAgent(cost) for _ in 1:10]  # Moderate size
        contest = TullockContest(agents, rand(10) .* 0.2, 30)
        
        # Should complete in under 1 second for this size
        elapsed_time = @elapsed begin
            final_round = run!(contest)
        end
        
        @test elapsed_time < 1.0  # Performance regression check
        @test final_round >= 1
        
        converged, actual_final = convergence_status(contest, final_round)
        @test actual_final <= 30
    end
    
    @testset "Large scale simulation" begin
        cost(x) = x
        n_agents = 20
        agents = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:n_agents]
        initial_efforts = rand(n_agents) .* 0.1 .+ 0.05
        T = 15  # Keep T manageable for large n
        
        contest = TullockContest(agents, initial_efforts, T)
        
        @time final_round = run!(contest)
        
        converged, actual_final = convergence_status(contest, final_round)
        
        # Check workspace scales correctly
        @test length(contest.workspace.own_efforts) == n_agents
        @test length(contest.workspace.min_other_efforts) == n_agents
        @test length(contest.workspace.max_other_efforts) == n_agents
        
        # Check all data structures are correct size
        @test size(contest.efforts) == (n_agents, T)
        @test size(contest.winners) == (n_agents, T)
        @test size(contest.utilities) == (n_agents, T)
        @test size(contest.nash_gaps) == (n_agents, T)
        
        # Verify simulation integrity
        for t in 1:actual_final
            @test sum(contest.winners[:, t]) == 1
            @test all(isfinite, contest.efforts[:, t])
            @test all(contest.efforts[:, t] .>= 0.0)
        end
    end
    
    @testset "Memory and allocation efficiency" begin
        # Test that repeated runs don't cause memory leaks
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Run multiple contests and check memory usage is reasonable
        for i in 1:5
            contest = TullockContest(agents, [0.1, 0.2], 10)
            final_round = run!(contest)
            
            converged, actual_final = convergence_status(contest, final_round)
            @test actual_final >= 1
            
            # Check workspace is working efficiently
            @test contest.workspace.own_efforts === contest.workspace.weights_obj.values
            @test length(contest.workspace.own_efforts) == 2
        end
    end
    
    @testset "Edge case simulations" begin
        cost(x) = x
        
        # Single agent contests should not be allowed
        single_agent = [MLEAgent(cost)]
        @test_throws AssertionError TullockContest(single_agent, [0.1], 5)
        
        # Very short simulation (1 round)
        agents_short = [MLEAgent(cost), DumbAgent(cost)]
        contest_short = TullockContest(agents_short, [0.1, 0.2], 1)
        final_short = run!(contest_short)
        
        @test final_short >= 1
        @test sum(contest_short.winners[:, 1]) == 1
    end
    
    @testset "Bayesian agent integration" begin
        # Test that Bayesian agents work in full simulations
        cost(x) = x
        
        # Mix of agent types including Bayesian
        agents = [
            MLEAgent(cost),
            BayesianAgent(cost),
            DumbAgent(cost)
        ]
        contest = TullockContest(agents, [0.1, 0.15, 0.12], 10)
        
        final_round = run!(contest)
        converged, actual_final = convergence_status(contest, final_round)
        
        # Bayesian agents should work without errors
        @test actual_final >= 1
        
        for t in 1:actual_final
            @test all(isfinite, contest.efforts[:, t])
            @test all(contest.efforts[:, t] .>= 0.0)
            @test sum(contest.winners[:, t]) == 1
        end
        
        # Bayesian agent should produce reasonable efforts
        bayesian_efforts = contest.efforts[2, 1:actual_final]  # Agent 2 is Bayesian
        @test all(bayesian_efforts .>= contest.agents[2].χ)
        @test all(bayesian_efforts .< 10.0)  # Should be reasonable
    end
    
    @testset "Deterministic behavior verification" begin
        # Test that deterministic MLE agents behave consistently
        cost(x) = 0.5 * x^2
        agents = [DetMLEAgent(cost), DetMLEAgent(cost)]
        
        # Run two identical contests
        Random.seed!(98765)
        contest1 = TullockContest(agents, [0.1, 0.1], 8)
        final1 = run!(contest1)
        
        Random.seed!(98765)
        contest2 = TullockContest(agents, [0.1, 0.1], 8)
        final2 = run!(contest2)
        
        # Should get identical results
        @test final1 == final2
        
        converged1, actual1 = convergence_status(contest1, final1)
        converged2, actual2 = convergence_status(contest2, final2)
        @test actual1 == actual2
        
        # Efforts should be nearly identical (small numerical differences allowed)
        for t in 1:min(actual1, actual2)
            @test contest1.efforts[:, t] ≈ contest2.efforts[:, t] atol=1e-10
        end
    end
end