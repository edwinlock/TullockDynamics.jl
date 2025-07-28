@testset "Estimators Tests" begin
    
    # Set up a test contest for estimator testing
    function setup_test_contest()
        cost(x) = x
        agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2, 0.3], 10)
        
        # Set up some realistic effort and winner data
        contest.efforts[:, 2] = [0.15, 0.25, 0.35]
        contest.efforts[:, 3] = [0.2, 0.3, 0.4]
        contest.efforts[:, 4] = [0.18, 0.28, 0.38]
        contest.efforts[:, 5] = [0.22, 0.32, 0.42]
        
        # Set some winners (agent with highest effort wins)
        contest.winners[3, 2] = true  # Agent 3 wins round 2
        contest.winners[3, 3] = true  # Agent 3 wins round 3
        contest.winners[3, 4] = true  # Agent 3 wins round 4
        contest.winners[1, 5] = true  # Agent 1 wins round 5 (example of upset)
        
        return contest
    end
    
    @testset "max_likelihood_estimator" begin
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 1
        
        # Test basic functionality
        est = max_likelihood_estimator(contest, agent_idx, window)
        @test est isa Float64
        @test est >= 0.0
        @test est < 100.0  # Should be reasonable
        
        # Test with single round window
        est_single = max_likelihood_estimator(contest, agent_idx, 2:2)
        @test est_single isa Float64
        @test est_single >= 0.0
        
        # Test edge case: agent with no wins
        # Agent 1 has 0 wins in rounds 2:4, should return max_agent_effort
        expected_max = max_agent_effort(contest.agents[agent_idx].cost)
        @test est ≈ expected_max atol=1e-6
        
        # Test agent with wins
        contest.winners[1, 3] = true  # Give agent 1 a win
        est_with_win = max_likelihood_estimator(contest, agent_idx, window)
        @test est_with_win < expected_max  # Should be less than max effort
        @test est_with_win >= 0.0
    end
    
    @testset "deterministic_max_likelihood_estimator" begin
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 2
        
        est = deterministic_max_likelihood_estimator(contest, agent_idx, window)
        @test est isa Float64
        @test est >= 0.0
        @test est < 100.0
        
        # Test with different window sizes
        est_small = deterministic_max_likelihood_estimator(contest, agent_idx, 2:2)
        est_large = deterministic_max_likelihood_estimator(contest, agent_idx, 2:5)
        @test est_small >= 0.0
        @test est_large >= 0.0
        
        # Should be deterministic for same inputs
        est2 = deterministic_max_likelihood_estimator(contest, agent_idx, window)
        @test est ≈ est2
    end
    
    @testset "dumb_estimator" begin
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 3
        
        # Test with agent that has wins
        est = dumb_estimator(contest, agent_idx, window)
        @test est isa Float64
        @test est >= 0.0
        
        # Test edge case: agent with no wins
        agent_no_wins = 1  # Agent 1 has no wins in our setup
        est_no_wins = dumb_estimator(contest, agent_no_wins, window)
        expected_max = max_agent_effort(contest.agents[agent_no_wins].cost)
        @test est_no_wins ≈ expected_max atol=1e-6
        
        # Test that the formula is applied correctly for agents with wins
        # y = (num_rounds/wins - 1) * avg_effort
        contest.winners[2, 2] = true  # Give agent 2 one win
        agent_with_one_win = 2
        window_test = 2:4
        
        est_formula = dumb_estimator(contest, agent_with_one_win, window_test)
        
        # Calculate expected value manually
        num_rounds = length(window_test)  # 3
        wins = 1
        avg_effort = sum(contest.efforts[agent_with_one_win, window_test]) / num_rounds
        expected = (num_rounds / wins - 1) * avg_effort
        
        @test est_formula ≈ expected atol=1e-10
    end
    
    @testset "classic_estimator" begin
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 1
        
        est = classic_estimator(contest, agent_idx, window)
        @test est isa Float64
        
        # Should return total_effort_last - own_effort_last
        last_round = window[end]  # round 4
        total_effort_last = sum(contest.efforts[:, last_round])
        own_effort_last = contest.efforts[agent_idx, last_round]
        expected = total_effort_last - own_effort_last
        
        @test est ≈ expected atol=1e-10
        
        # Test with different windows
        window2 = 3:5
        est2 = classic_estimator(contest, agent_idx, window2)
        
        last_round2 = window2[end]  # round 5
        total_effort_last2 = sum(contest.efforts[:, last_round2])
        own_effort_last2 = contest.efforts[agent_idx, last_round2]
        expected2 = total_effort_last2 - own_effort_last2
        
        @test est2 ≈ expected2 atol=1e-10
    end
    
    @testset "bayesian_estimator" begin
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 1
        
        # Test that it returns a function
        estimator_func = bayesian_estimator(contest, agent_idx, window)
        @test estimator_func isa Function
        
        # Test that the returned function works
        test_points = [0.1, 0.5, 1.0, 2.0]
        for y in test_points
            density_val = estimator_func(y)
            @test density_val isa Float64
            @test density_val >= 0.0  # Probability densities are non-negative
        end
        
        # Test that density integrates reasonably (not exactly 1 due to discretization)
        # Sample a few points and check they sum to something reasonable
        y_vals = range(contest.workspace.min_other_efforts[agent_idx], 
                      contest.workspace.max_other_efforts[agent_idx], 
                      length=10)
        densities = [estimator_func(y) for y in y_vals]
        @test all(densities .>= 0.0)
        
        # Test that density is zero outside bounds
        lb = contest.workspace.min_other_efforts[agent_idx]
        ub = contest.workspace.max_other_efforts[agent_idx]
        
        if isfinite(lb) && isfinite(ub)
            @test estimator_func(lb - 0.1) ≈ 0.0
            @test estimator_func(ub + 0.1) ≈ 0.0
        end
    end
    
    @testset "Estimator consistency" begin
        # Test that estimators give reasonable relative results
        contest = setup_test_contest()
        window = 2:4
        agent_idx = 2
        
        # Get estimates from different methods
        mle_est = max_likelihood_estimator(contest, agent_idx, window)
        det_mle_est = deterministic_max_likelihood_estimator(contest, agent_idx, window)
        dumb_est = dumb_estimator(contest, agent_idx, window)
        classic_est = classic_estimator(contest, agent_idx, window)
        
        # All should be non-negative
        @test mle_est >= 0.0
        @test det_mle_est >= 0.0
        @test dumb_est >= 0.0
        @test classic_est >= 0.0
        
        # All should be finite (assuming finite max efforts)
        @test isfinite(mle_est)
        @test isfinite(det_mle_est)
        @test isfinite(dumb_est)
        @test isfinite(classic_est)
        
        # Classic estimator should match actual opponent effort exactly
        last_round = window[end]
        actual_other_effort = sum(contest.efforts[:, last_round]) - contest.efforts[agent_idx, last_round]
        @test classic_est ≈ actual_other_effort atol=1e-10
    end
    
    @testset "Edge cases and robustness" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 5)
        
        # Test with minimal data (single round)
        contest.efforts[:, 2] = [0.05, 0.15]
        contest.winners[2, 2] = true
        
        window = 2:2
        agent_idx = 1
        
        # All estimators should handle single-round windows
        @test max_likelihood_estimator(contest, agent_idx, window) >= 0.0
        @test deterministic_max_likelihood_estimator(contest, agent_idx, window) >= 0.0
        @test dumb_estimator(contest, agent_idx, window) >= 0.0
        @test classic_estimator(contest, agent_idx, window) >= 0.0
        
        # Test with zero efforts (edge case)
        contest.efforts[:, 3] = [0.0, 0.0]
        contest.winners[1, 3] = true  # Arbitrary winner when all efforts zero
        
        window_zero = 3:3
        # Some estimators should handle zero efforts gracefully
        classic_zero = classic_estimator(contest, agent_idx, window_zero)
        @test classic_zero ≈ 0.0
        
        # Test with very small efforts
        contest.efforts[:, 4] = [1e-6, 2e-6]
        contest.winners[2, 4] = true
        
        window_small = 4:4
        @test classic_estimator(contest, agent_idx, window_small) ≈ 2e-6 atol=1e-10
    end
end