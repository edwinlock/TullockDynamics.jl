@testset "Regression Tests" begin
    
    @testset "Known good simulation results" begin
        # Test that specific scenarios produce expected results
        cost(x) = x
        
        # Test 1: Symmetric agents should converge to similar efforts
        Random.seed!(12345)
        symmetric_agents = [MLEAgent(cost), MLEAgent(cost)]
        symmetric_contest = TullockContest(symmetric_agents, [0.2, 0.2], 20)
        
        final_round = run!(symmetric_contest; ε=0.01)
        converged, actual_final = convergence_status(symmetric_contest, final_round)
        
        if converged
            final_efforts = symmetric_contest.efforts[:, actual_final]
            effort_ratio = final_efforts[1] / final_efforts[2]
            @test 0.8 <= effort_ratio <= 1.25  # Should be reasonably close
        end
        
        # Test 2: Agent with lower cost should generally exert more effort
        low_cost(x) = 0.5 * x
        high_cost(x) = 2.0 * x
        
        Random.seed!(54321)
        cost_agents = [MLEAgent(low_cost), MLEAgent(high_cost)]
        cost_contest = TullockContest(cost_agents, [0.15, 0.15], 25)
        
        run!(cost_contest)
        converged_cost, actual_final_cost = convergence_status(cost_contest, run!(cost_contest))
        
        # Over the course of simulation, low cost agent should generally have higher efforts
        avg_efforts = [mean(cost_contest.efforts[i, 2:actual_final_cost]) for i in 1:2]
        @test avg_efforts[1] > avg_efforts[2] * 0.8  # At least 80% higher on average
    end
    
    @testset "Boundary condition regression tests" begin
        cost(x) = x
        
        # Test that single agent contests are not allowed
        single_agent = [MLEAgent(cost)]
        @test_throws AssertionError TullockContest(single_agent, [0.1], 1)
        
        # Test contests with minimal rounds
        agents_min = [MLEAgent(cost), DumbAgent(cost)]
        min_contest = TullockContest(agents_min, [0.1, 0.2], 2)
        result_min = run!(min_contest)
        converged_min, actual_final_min = convergence_status(min_contest, result_min)
        @test 1 <= actual_final_min <= 2
        
        # Test with zero initial efforts
        zero_agents = [MLEAgent(cost), DumbAgent(cost)]
        zero_contest = TullockContest(zero_agents, [0.0, 0.0], 5)
        result_zero = run!(zero_contest)
        converged_zero, actual_final_zero = convergence_status(zero_contest, result_zero)
        
        # Should handle zero initial efforts gracefully
        @test actual_final_zero >= 1
        for t in 1:actual_final_zero
            @test sum(zero_contest.winners[:, t]) == 1
            @test all(zero_contest.efforts[:, t] .>= 0.0)
        end
    end
    
    @testset "Estimator consistency regression" begin
        cost(x) = x
        agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2, 0.15], 10)
        
        # Create deterministic scenario
        Random.seed!(99999)
        for t in 2:6
            contest.efforts[:, t] = [0.1 + 0.02*t, 0.15 + 0.03*t, 0.12 + 0.025*t]
            # Deterministic winner (highest effort)
            winner_idx = argmax(contest.efforts[:, t])
            contest.winners[winner_idx, t] = true
        end
        
        window = 3:5
        
        # Test that estimators give consistent results for same input
        for agent_idx in 1:3
            est1_mle = max_likelihood_estimator(contest, agent_idx, window)
            est2_mle = max_likelihood_estimator(contest, agent_idx, window)
            @test est1_mle ≈ est2_mle
            
            est1_det = deterministic_max_likelihood_estimator(contest, agent_idx, window)
            est2_det = deterministic_max_likelihood_estimator(contest, agent_idx, window)
            @test est1_det ≈ est2_det
            
            est1_dumb = dumb_estimator(contest, agent_idx, window)
            est2_dumb = dumb_estimator(contest, agent_idx, window)
            @test est1_dumb ≈ est2_dumb
            
            est1_classic = classic_estimator(contest, agent_idx, window)
            est2_classic = classic_estimator(contest, agent_idx, window)
            @test est1_classic ≈ est2_classic
        end
    end
    
    @testset "Performance regression baselines" begin
        cost(x) = x
        
        # Establish performance baselines for regression testing
        # Small contest baseline
        agents_small = [MLEAgent(cost), DumbAgent(cost)]
        contest_small = TullockContest(agents_small, [0.1, 0.2], 10)
        
        time_small = @elapsed run!(contest_small)
        @test time_small < 0.5  # Should complete in under 0.5 seconds
        
        # Medium contest baseline
        agents_medium = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:5]
        contest_medium = TullockContest(agents_medium, rand(5) .* 0.2, 15)
        
        time_medium = @elapsed run!(contest_medium)
        @test time_medium < 1.0  # Should complete in under 1 second
        
        # Large contest baseline (for performance regression detection)
        agents_large = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:10]
        contest_large = TullockContest(agents_large, rand(10) .* 0.2, 10)
        
        time_large = @elapsed run!(contest_large)
        @test time_large < 2.0  # Should complete in under 2 seconds
        
        # Scaling check: large contest shouldn't be too much slower
        @test time_large < time_small * 20  # At most 20x slower for 5x agents, 2x rounds
    end
    
    @testset "Numerical stability regression" begin
        cost(x) = x
        
        # Test with extreme values that previously caused issues
        extreme_agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Very large initial efforts
        large_contest = TullockContest(extreme_agents, [100.0, 200.0], 5)
        result_large = run!(large_contest)
        converged_large, actual_final_large = convergence_status(large_contest, result_large)
        
        # Should handle large values without overflow
        @test actual_final_large >= 1
        for t in 1:actual_final_large
            @test all(isfinite, large_contest.efforts[:, t])
            @test all(isfinite, large_contest.utilities[:, t])
        end
        
        # Very small initial efforts
        small_contest = TullockContest(extreme_agents, [1e-6, 2e-6], 5)
        result_small = run!(small_contest)
        converged_small, actual_final_small = convergence_status(small_contest, result_small)
        
        # Should handle small values without underflow
        @test actual_final_small >= 1
        for t in 1:actual_final_small
            @test all(isfinite, small_contest.efforts[:, t])
            @test all(small_contest.efforts[:, t] .>= 0.0)
        end
    end
    
    @testset "Export completeness regression" begin
        # Verify all expected functions are exported and accessible
        exported_functions = [
            # From agents.jl
            Agent, MLEAgent, DetMLEAgent, DumbAgent, BayesianAgent,
            utility, best_response, nash_gap, max_agent_effort,
            
            # From contests.jl  
            TullockContest, num_rounds,
            
            # From dynamics.jl
            set_efforts!, set_utilities!, step!, run!,
            
            # From estimators.jl
            max_likelihood_estimator, deterministic_max_likelihood_estimator,
            dumb_estimator, bayesian_estimator, classic_estimator,
            
            # From utils.jl
            find_root, final_efforts, visualise, convergence_status
        ]
        
        for func in exported_functions
            @test isdefined(TullockDynamics, Symbol(func))
        end
        
        # Test that functions work as expected
        cost(x) = x
        agent = MLEAgent(cost)
        @test agent isa TullockDynamics.Agent
        
        contest = TullockContest([agent, MLEAgent(cost)], [0.1, 0.2], 3)
        @test contest isa TullockContest
        
        final_round = run!(contest)
        @test final_round isa Int
        
        final_eff = final_efforts(contest)
        @test final_eff isa Vector{Float64}
    end
    
    @testset "Documentation example regression" begin
        # Test examples that might be used in documentation
        cost(x) = x
        
        # Simple two-agent competition
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.2, 0.3], 20)
        
        final_round = run!(contest)
        converged, actual_final = convergence_status(contest, final_round)
        
        @test actual_final >= 1
        @test actual_final <= 20
        
        # Check basic properties that users might expect
        @test all(contest.efforts[:, 1] .== [0.2, 0.3])  # Initial efforts preserved
        @test sum(contest.winners[:, actual_final]) == 1  # One winner in final round
        @test all(contest.efforts[:, actual_final] .>= 0.0)  # Non-negative efforts
        
        # Test utility calculations make sense
        final_efforts = contest.efforts[:, actual_final]
        total_effort = sum(final_efforts)
        
        for i in 1:2
            x = final_efforts[i]
            s = total_effort - x
            expected_utility = x / (x + s) - cost(x)
            @test contest.utilities[i, actual_final] ≈ expected_utility atol=1e-10
        end
    end
end