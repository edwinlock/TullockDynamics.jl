@testset "Dynamics Tests" begin
    
    function setup_contest_for_dynamics()
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost), DetMLEAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2, 0.15], 10)
        return contest
    end
    
    @testset "set_utilities!" begin
        contest = setup_contest_for_dynamics()
        
        # Set some effort values for round 2
        contest.efforts[:, 2] = [0.2, 0.3, 0.25]
        
        # Call set_utilities!
        TullockDynamics.set_utilities!(contest, 2)
        
        # Check that utilities are computed and finite
        @test all(isfinite, contest.utilities[:, 2])
        
        # Manually verify utility calculation for first agent
        x = contest.efforts[1, 2]  # 0.2
        total_effort = sum(contest.efforts[:, 2])  # 0.75
        s = total_effort - x  # 0.55
        expected_utility = x / (x + s) - contest.agents[1].cost(x)  # 0.2/0.75 - 0.2
        @test contest.utilities[1, 2] ≈ expected_utility atol=1e-10
        
        # Check workspace is updated correctly
        @test contest.workspace.total_effort ≈ total_effort
        @test contest.workspace.all_efforts ≈ contest.efforts[:, 2]
        
        # Test utilities are reasonable (between -10 and 1)
        @test all(-10.0 .< contest.utilities[:, 2] .< 1.0)
    end
    
    @testset "set_nash_gap!" begin
        contest = setup_contest_for_dynamics()
        
        # Set efforts for round 3
        contest.efforts[:, 3] = [0.15, 0.35, 0.2]
        
        # Call set_nash_gap!
        TullockDynamics.set_nash_gap!(contest, 3)
        
        # Check that Nash gaps are computed and non-negative
        @test all(>=(0.0), contest.nash_gaps[:, 3])
        @test all(isfinite, contest.nash_gaps[:, 3])
        
        # Manually verify Nash gap calculation for first agent
        x = contest.efforts[1, 3]
        total_effort = sum(contest.efforts[:, 3])
        s = total_effort - x
        expected_gap = nash_gap(contest.agents[1], x, s)
        @test contest.nash_gaps[1, 3] ≈ expected_gap atol=1e-10
        
        # Check workspace is used correctly
        @test contest.workspace.total_effort ≈ total_effort
        @test contest.workspace.all_efforts ≈ contest.efforts[:, 3]
        
        # Nash gaps should typically be small for reasonable strategies
        @test all(contest.nash_gaps[:, 3] .< 10.0)
    end
    
    @testset "set_efforts!" begin
        contest = setup_contest_for_dynamics()
        
        # Set initial data for round 2 estimation
        contest.efforts[:, 2] = [0.12, 0.22, 0.18]
        contest.winners[2, 2] = true  # Agent 2 wins round 2
        
        # Save original efforts for comparison
        original_efforts_1 = copy(contest.efforts[:, 1])
        original_efforts_2 = copy(contest.efforts[:, 2])
        
        # Test assertion for t < 2
        @test_throws AssertionError TullockDynamics.set_efforts!(contest, 1)
        
        # Call set_efforts! for round 3
        TullockDynamics.set_efforts!(contest, 3)
        
        # Check that efforts are updated for round 3
        @test all(contest.efforts[:, 3] .>= 0.0)
        @test all(isfinite, contest.efforts[:, 3])
        
        # Previous rounds should be unchanged
        @test contest.efforts[:, 1] == original_efforts_1
        @test contest.efforts[:, 2] == original_efforts_2
        
        # Efforts should be reasonable (bounded by agent constraints)
        for i in eachindex(contest.agents)
            @test contest.efforts[i, 3] >= contest.agents[i].χ
            if isfinite(contest.agents[i].max_effort)
                @test contest.efforts[i, 3] <= contest.agents[i].max_effort
            end
        end
    end
    
    @testset "step!" begin
        contest = setup_contest_for_dynamics()
        
        # Test step for t=1 (no effort updates, just utilities/gaps/winner)
        TullockDynamics.step!(contest, 1)
        
        # Check that exactly one winner is selected
        @test sum(contest.winners[:, 1]) == 1
        @test any(contest.winners[:, 1])
        
        # Check that utilities and nash gaps are computed
        @test all(isfinite, contest.utilities[:, 1])
        @test all(>=(0.0), contest.nash_gaps[:, 1])
        
        # Save state before step 2
        original_efforts_1 = copy(contest.efforts[:, 1])
        
        # Test step for t=2 (includes effort updates)
        TullockDynamics.step!(contest, 2)
        
        # Check winner selection
        @test sum(contest.winners[:, 2]) == 1
        @test any(contest.winners[:, 2])
        
        # Check utilities and nash gaps
        @test all(isfinite, contest.utilities[:, 2])
        @test all(>=(0.0), contest.nash_gaps[:, 2])
        
        # Efforts for round 1 should be unchanged
        @test contest.efforts[:, 1] == original_efforts_1
        
        # Efforts for round 2 should be updated and valid
        @test all(contest.efforts[:, 2] .>= 0.0)
        @test all(isfinite, contest.efforts[:, 2])
    end
    
    @testset "run!" begin
        contest = setup_contest_for_dynamics()
        
        # Test basic run
        final_round = run!(contest)
        
        # Should return a valid round number
        @test final_round >= 1
        @test final_round <= num_rounds(contest) + 1  # Allow T+1 for non-convergence
        
        # Determine actual final round for data access
        converged, actual_final = convergence_status(contest, final_round)
        
        # Check that all rounds up to actual_final have valid data
        for t in 1:actual_final
            # Exactly one winner per round
            @test sum(contest.winners[:, t]) == 1
            
            # Valid utilities and nash gaps
            @test all(isfinite, contest.utilities[:, t])
            @test all(>=(0.0), contest.nash_gaps[:, t])
            
            # Valid efforts
            @test all(contest.efforts[:, t] .>= 0.0)
            @test all(isfinite, contest.efforts[:, t])
        end
        
        # Test convergence detection
        if converged
            @test final_round <= num_rounds(contest)
            @test final_round == actual_final
            
            # The Nash gap at convergence should be small (if ε was set)
            # Since default ε = -1, this might not converge early
        else
            @test final_round == num_rounds(contest) + 1
            @test actual_final == num_rounds(contest)
        end
    end
    
    @testset "run! with convergence" begin
        contest = setup_contest_for_dynamics()
        
        # Test with convergence threshold
        ε = 0.1
        final_round = run!(contest; ε=ε)
        
        converged, actual_final = convergence_status(contest, final_round)
        
        if converged
            # If converged, the total Nash gap should be ≤ ε
            total_gap = nash_gap(contest, actual_final)
            @test total_gap <= ε
            @test final_round < num_rounds(contest)
        else
            # If not converged, should have run full duration
            @test final_round == num_rounds(contest) + 1
            @test actual_final == num_rounds(contest)
        end
    end
    
    @testset "run! reproducibility" begin
        # Test that runs with same random seed give same results
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Run 1
        Random.seed!(12345)
        contest1 = TullockContest(agents, [0.1, 0.2], 5)
        final1 = run!(contest1)
        
        # Run 2 with same seed
        Random.seed!(12345)  
        contest2 = TullockContest(agents, [0.1, 0.2], 5)
        final2 = run!(contest2)
        
        # Should get identical results
        @test final1 == final2
        
        # Efforts should be identical
        converged1, actual1 = convergence_status(contest1, final1)
        converged2, actual2 = convergence_status(contest2, final2)
        @test actual1 == actual2
        
        for t in 1:min(actual1, actual2)
            @test contest1.efforts[:, t] ≈ contest2.efforts[:, t] atol=1e-12
            @test contest1.winners[:, t] == contest2.winners[:, t]
        end
    end
    
    @testset "Workspace efficiency" begin
        contest = setup_contest_for_dynamics()
        
        # Run a few steps and check workspace is reused
        TullockDynamics.step!(contest, 1)
        workspace_after_1 = contest.workspace
        
        TullockDynamics.step!(contest, 2)
        workspace_after_2 = contest.workspace
        
        # Should be the same object (not reallocated)
        @test workspace_after_1 === workspace_after_2
        
        # Buffers should be reused
        @test workspace_after_1.all_efforts === workspace_after_2.all_efforts
        @test workspace_after_1.weights_obj === workspace_after_2.weights_obj
        
        # Values should be updated appropriately
        @test workspace_after_2.total_effort > 0.0
        @test sum(workspace_after_2.all_efforts) ≈ workspace_after_2.total_effort
    end
    
    @testset "Multi-agent dynamics" begin
        # Test with larger number of agents
        cost(x) = x
        n_agents = 8
        agents = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:n_agents]
        initial_efforts = rand(n_agents) * 0.2 .+ 0.05  # Random efforts 0.05-0.25
        
        contest = TullockContest(agents, initial_efforts, 15)
        final_round = run!(contest)
        
        converged, actual_final = convergence_status(contest, final_round)
        
        # Check all agents are handled correctly
        for t in 1:actual_final
            @test sum(contest.winners[:, t]) == 1  # One winner
            @test all(contest.efforts[:, t] .>= 0.0)  # Valid efforts
            @test all(isfinite, contest.utilities[:, t])  # Valid utilities
        end
        
        # Check workspace dimensions match
        @test length(contest.workspace.all_efforts) == n_agents
        @test length(contest.workspace.min_other_bounds) == n_agents
        @test length(contest.workspace.max_other_bounds) == n_agents
    end
end