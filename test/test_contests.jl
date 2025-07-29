@testset "Contests Tests" begin
    
    @testset "TullockContest construction" begin
        cost(x) = x
        agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]
        initial_efforts = [0.1, 0.2, 0.3]
        T = 15
        
        contest = TullockContest(agents, initial_efforts, T)
        
        # Test basic structure
        @test length(contest.agents) == 3
        @test size(contest.efforts) == (3, 15)
        @test size(contest.winners) == (3, 15)
        @test size(contest.utilities) == (3, 15)
        @test size(contest.nash_gaps) == (3, 15)
        
        # Test initial efforts are set correctly
        @test contest.efforts[:, 1] == initial_efforts
        
        # Test matrices are properly initialized (zeros/false)
        @test all(contest.efforts[:, 2:end] .== 0.0)
        @test all(contest.winners[:, :] .== false)
        @test all(contest.utilities[:, :] .== 0.0)
        @test all(contest.nash_gaps[:, :] .== 0.0)
        
        # Test workspace is properly initialized
        @test contest.workspace isa TullockDynamics.ContestWorkspace
        @test length(contest.efforts[:,1]) == 3
        @test contest.workspace.min_total_efforts > 0.0
        @test contest.workspace.max_total_efforts >= contest.workspace.min_total_efforts
        @test length(contest.workspace.min_other_efforts) == 3
        @test length(contest.workspace.max_other_efforts) == 3
        @test all(contest.workspace.total_efforts .== 0.0)
    end
    
    @testset "TullockContest validation" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Test mismatched agent and effort vector lengths
        @test_throws AssertionError TullockContest(agents, [0.1], 10)  # Too few efforts
        @test_throws AssertionError TullockContest(agents, [0.1, 0.2, 0.3], 10)  # Too many efforts
        
        # Test invalid number of rounds
        @test_throws AssertionError TullockContest(agents, [0.1, 0.2], 0)
        @test_throws AssertionError TullockContest(agents, [0.1, 0.2], -5)
        
        # Test valid edge cases
        contest_1_round = TullockContest(agents, [0.1, 0.2], 1)
        @test size(contest_1_round.efforts) == (2, 1)
        
        # Single agent contests should not be allowed
        @test_throws AssertionError TullockContest([agents[1]], [0.1], 5)
    end
    
    @testset "num_rounds function" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        contest_10 = TullockContest(agents, [0.1, 0.2], 10)
        @test num_rounds(contest_10) == 10
        
        contest_1 = TullockContest(agents, [0.1, 0.2], 1)
        @test num_rounds(contest_1) == 1
        
        contest_100 = TullockContest(agents, [0.1, 0.2], 100)
        @test num_rounds(contest_100) == 100
    end
    
    @testset "nash_gap function (contest-level)" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 5)
        
        # Set some nash gap values
        contest.nash_gaps[:, 3] = [0.1, 0.2]
        
        total_gap = nash_gap(contest, 3)
        @test total_gap ≈ 0.3
        
        # Test with zeros
        contest.nash_gaps[:, 4] = [0.0, 0.0]
        total_gap = nash_gap(contest, 4)
        @test total_gap ≈ 0.0
        
        # Test with minimal case (2 agents)
        contest_min = TullockContest(agents, [0.1, 0.2], 5)
        contest_min.nash_gaps[1, 2] = 0.5
        contest_min.nash_gaps[2, 2] = 0.3
        total_gap = nash_gap(contest_min, 2)
        @test total_gap ≈ 0.8
    end
    
    @testset "ContestWorkspace pre-computed values" begin
        cost1(x) = x      # χ = 0.01, max_effort = 1.0 (since cost(1) = 1)
        cost2(x) = x^2    # χ = 0.01, max_effort = 1.0 (since cost(1) = 1)
        
        agent1 = MLEAgent(cost1; χ=0.02)  # min = 0.02
        agent2 = DumbAgent(cost2; χ=0.03)  # min = 0.03
        agents = [agent1, agent2]
        
        contest = TullockContest(agents, [0.1, 0.2], 10)
        workspace = contest.workspace
        
        # Test pre-computed total bounds
        expected_min_total = 0.02 + 0.03  # sum of χ values
        expected_max_total = 1.0 + 1.0    # sum of max_effort values
        @test workspace.min_total_efforts ≈ expected_min_total
        @test workspace.max_total_efforts ≈ expected_max_total
        
        # Test per-agent bounds
        # Agent 1: min_other = min_total - χ[1] = 0.05 - 0.02 = 0.03, max_other = max_total - max_effort[1] = 2.0 - 1.0 = 1.0
        @test workspace.min_other_efforts[1] ≈ 0.03
        @test workspace.max_other_efforts[1] ≈ 1.0
        
        # Agent 2: min_other = min_total - χ[2] = 0.05 - 0.03 = 0.02, max_other = max_total - max_effort[2] = 2.0 - 1.0 = 1.0
        @test workspace.min_other_efforts[2] ≈ 0.02  
        @test workspace.max_other_efforts[2] ≈ 1.0
    end
    
    @testset "ContestWorkspace with finite max_effort" begin
        cost(x) = x
        
        # Create agents with finite max efforts
        agent1 = Agent(max_likelihood_estimator, cost, t->1.0, t->1.0, 0.01, 2.0, t->1:t-1)
        agent2 = Agent(dumb_estimator, cost, t->1.0, t->1.0, 0.02, 3.0, t->1:t-1)
        agents = [agent1, agent2]
        
        contest = TullockContest(agents, [0.1, 0.2], 5)
        workspace = contest.workspace
        
        # Test bounds with finite max efforts
        expected_min_total = 0.01 + 0.02  # 0.03
        expected_max_total = 2.0 + 3.0    # 5.0
        @test workspace.min_total_efforts ≈ expected_min_total
        @test workspace.max_total_efforts ≈ expected_max_total
        
        # Agent 1: min_other = 0.03 - 0.01 = 0.02
        #          max_other = 5.0 - 2.0 = 3.0
        @test workspace.min_other_efforts[1] ≈ 0.02
        @test workspace.max_other_efforts[1] ≈ 3.0
        
        # Agent 2: min_other = 0.03 - 0.02 = 0.01
        #          max_other = 5.0 - 3.0 = 2.0
        @test workspace.min_other_efforts[2] ≈ 0.01
        @test workspace.max_other_efforts[2] ≈ 2.0
    end
end