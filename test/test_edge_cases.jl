@testset "Edge Cases and Error Handling" begin
    
    @testset "Agent edge cases" begin
        cost(x) = x
        
        # Test agent with zero minimum effort
        agent_zero_chi = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.0, max_agent_effort(cost), t->1:t-1)
        @test agent_zero_chi.χ == 0.0
        
        # Test utility with zero efforts
        @test isnan(utility(agent_zero_chi, 0.0, 0.0))  # 0/0 results in NaN in Julia
        @test utility(agent_zero_chi, 0.0, 1.0) ≈ 0.0
        
        # Test best response with small but non-zero opponent effort (avoid edge case)
        br = best_response(agent_zero_chi, 0.001)
        @test br >= 0.0
        @test isfinite(br)
        
        # Test Nash gap with small but non-zero efforts (avoid edge case)
        gap = nash_gap(agent_zero_chi, 0.001, 0.001)
        @test gap >= 0.0
        @test isfinite(gap)
    end
    
    @testset "Cost function edge cases" begin
        # Test with constant cost function
        const_cost(x) = 1.0
        agent = MLEAgent(const_cost)
        
        # Best response should still work
        br = best_response(agent, 1.0)
        @test br >= agent.χ
        @test isfinite(br)
        
        # Test with very steep cost function
        steep_cost(x) = x^10
        agent_steep = MLEAgent(steep_cost)
        br_steep = best_response(agent_steep, 1.0)
        @test br_steep >= agent_steep.χ
        @test br_steep < 10.0  # Should be reasonable
        
        # Test with discontinuous cost (step function)
        step_cost(x) = x < 0.5 ? 0.1 : 10.0
        agent_step = MLEAgent(step_cost)
        br_step = best_response(agent_step, 1.0)
        @test br_step >= agent_step.χ
    end
    
    @testset "Contest construction edge cases" begin
        cost(x) = x
        
        # Single agent contest should not be allowed
        single_agent = [MLEAgent(cost)]
        @test_throws AssertionError TullockContest(single_agent, [0.1], 1)
        
        # Very large initial efforts
        agents = [MLEAgent(cost), DumbAgent(cost)]
        large_efforts = [100.0, 200.0]
        contest_large = TullockContest(agents, large_efforts, 3)
        @test contest_large.efforts[:, 1] == large_efforts
        
        # Very small initial efforts
        small_efforts = [1e-10, 2e-10]
        contest_small = TullockContest(agents, small_efforts, 3)
        @test contest_small.efforts[:, 1] == small_efforts
        
        # Zero initial efforts
        zero_efforts = [0.0, 0.0]
        contest_zero = TullockContest(agents, zero_efforts, 3)
        @test contest_zero.efforts[:, 1] == zero_efforts
    end
    
    @testset "Estimator edge cases" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 5)
        
        # Set up scenario with all zeros
        contest.efforts[:, 2] = [0.0, 0.0]
        contest.efforts[:, 3] = [0.0, 0.0]
        contest.winners[1, 2] = true  # Arbitrary winner when efforts are zero
        contest.winners[2, 3] = true
        
        window = 2:3
        
        # Test estimators with zero efforts
        classic_est = classic_estimator(contest, 1, window)
        @test classic_est ≈ 0.0
        
        # Test with single round window using non-zero efforts
        contest.efforts[:, 2] = [0.01, 0.02]  # Use small but non-zero efforts
        single_window = 2:2
        for estimator_func in [max_likelihood_estimator, deterministic_max_likelihood_estimator, 
                              dumb_estimator, classic_estimator]
            est = estimator_func(contest, 1, single_window)
            @test isfinite(est)
            @test est >= 0.0
        end
        
        # Test Bayesian estimator with zero losses
        contest.winners[1, 2] = true  # Agent 1 wins all rounds (no losses)
        contest.winners[1, 3] = true
        contest.efforts[:, 2] = [0.1, 0.05]  # Non-zero efforts
        contest.efforts[:, 3] = [0.15, 0.08]
        
        bayesian_func = bayesian_estimator(contest, 1, window)
        @test bayesian_func isa Function
        
        # Test density at a few points
        test_y = contest.workspace.min_other_bounds[1] + 0.1
        if isfinite(test_y)
            density = bayesian_func(test_y)
            @test density >= 0.0
            @test isfinite(density)
        end
    end
    
    @testset "Dynamics with extreme parameters" begin
        cost(x) = x
        
        # Test with agents that never update (p(t) = 0)
        never_update_agent = Agent(max_likelihood_estimator, cost, t->0.0, t->1.0, 0.01, max_agent_effort(cost), t->1:t-1)
        agents_no_update = [never_update_agent, MLEAgent(cost)]
        contest_no_update = TullockContest(agents_no_update, [0.1, 0.2], 5)
        
        final_round = run!(contest_no_update)
        converged, actual_final = convergence_status(contest_no_update, final_round)
        
        # First agent's efforts should remain constant
        @test all(contest_no_update.efforts[1, :] .≈ 0.1)
        
        # Test with agents that always update (p(t) = 1)
        always_update_agent = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.01, max_agent_effort(cost), t->1:t-1)
        agents_always_update = [always_update_agent, always_update_agent]
        contest_always_update = TullockContest(agents_always_update, [0.1, 0.2], 3)
        
        final_round_always = run!(contest_always_update)
        converged_always, actual_final_always = convergence_status(contest_always_update, final_round_always)
        
        # Efforts should change after round 1
        @test !all(contest_always_update.efforts[:, 2] .≈ [0.1, 0.2])
        
        # Test with very slow adaptation (α(t) = 0.01)
        slow_agent = Agent(max_likelihood_estimator, cost, t->1.0, t->0.01, 0.01, max_agent_effort(cost), t->1:t-1)
        agents_slow = [slow_agent, slow_agent]
        contest_slow = TullockContest(agents_slow, [0.1, 0.2], 5)
        
        run!(contest_slow)
        
        # Changes should be gradual
        diff_slow = abs(contest_slow.efforts[1, 3] - contest_slow.efforts[1, 2])
        @test diff_slow < 0.1  # Should change slowly
    end
    
    @testset "Memory window edge cases" begin
        cost(x) = x
        
        # Agent with very long memory window
        long_memory_agent = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.01, max_agent_effort(cost), t->1:t-1)
        
        # Agent with single-round memory
        short_memory_agent = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.01, max_agent_effort(cost), t->max(1,t-1):t-1)
        
        agents_memory = [long_memory_agent, short_memory_agent]
        contest_memory = TullockContest(agents_memory, [0.1, 0.2], 8)
        
        final_round = run!(contest_memory)
        converged, actual_final = convergence_status(contest_memory, final_round)
        
        # Should complete without errors
        @test actual_final >= 1
        for t in 1:actual_final
            @test all(isfinite, contest_memory.efforts[:, t])
        end
    end
    
    @testset "Numerical stability" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Test with very large number of rounds
        contest_long = TullockContest(agents, [0.1, 0.2], 100)
        final_round = run!(contest_long)
        converged, actual_final = convergence_status(contest_long, final_round)
        
        # Should not have NaN or Inf values
        for t in 1:actual_final
            @test all(isfinite, contest_long.efforts[:, t])
            @test all(isfinite, contest_long.utilities[:, t])
            @test all(isfinite, contest_long.nash_gaps[:, t])
        end
        
        # Test convergence detection with very small epsilon
        tiny_epsilon = 1e-12
        contest_precise = TullockContest(agents, [0.1, 0.2], 20)
        final_round_precise = run!(contest_precise; ε=tiny_epsilon)
        
        converged_precise, actual_final_precise = convergence_status(contest_precise, final_round_precise)
        
        if converged_precise
            total_gap = nash_gap(contest_precise, actual_final_precise)
            @test total_gap <= tiny_epsilon
        end
    end
    
    @testset "Workspace bounds verification" begin
        cost(x) = x
        
        # Create agents with different χ and max_effort values
        agent1 = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.05, 5.0, t->1:t-1)
        agent2 = Agent(dumb_estimator, cost, t->1.0, t->1.0, 0.02, 3.0, t->1:t-1)
        agents = [agent1, agent2]
        
        contest = TullockContest(agents, [0.1, 0.2], 5)
        workspace = contest.workspace
        
        # Verify bounds calculations
        expected_min_total = 0.05 + 0.02  # 0.07
        expected_max_total = 5.0 + 3.0    # 8.0
        
        @test workspace.min_total_efforts ≈ expected_min_total
        @test workspace.max_total_efforts ≈ expected_max_total
        
        # Check individual bounds
        # Agent 1: min_other = max(0.001, 0.07 - 5.0) = 0.001, max_other = max(0.01, 8.0 - 0.05) = 7.95
        @test workspace.min_other_bounds[1] ≈ 0.001
        @test workspace.max_other_bounds[1] ≈ 7.95
        
        # Agent 2: min_other = max(0.001, 0.07 - 3.0) = 0.001, max_other = max(0.01, 8.0 - 0.02) = 7.98
        @test workspace.min_other_bounds[2] ≈ 0.001
        @test workspace.max_other_bounds[2] ≈ 7.98
    end
end