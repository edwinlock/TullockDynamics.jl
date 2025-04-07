module TullockApp

using Dash
using DashBootstrapComponents
using PlotlyJS

# Load my TullockDynamics package
using Pkg; Pkg.activate("../TullockDynamics.jl")
using TullockDynamics

# Enter 9999 in the port field in the deploy form when deploying on JuliaHub
const PORT = 9999

const DEV_MODE = haskey(ENV, "VSCODE_PROXY_URI")

# some more constants
const max_num_agents = 10
const max_num_rounds = 50000
const A = [0.1 ; 0.15]'

function run_app(host="0.0.0.0", port=PORT)
    @info("Initializing dash...")
    app = dash(;
        external_stylesheets=[dbc_themes.BOOTSTRAP],
        requests_pathname_prefix=(DEV_MODE ? "/proxy/$(string(port))/" : "/"),
    )


    navbar = dbc_navbarsimple(
        brand="Tullock Dynamics Simulations",
        brand_href="#",
        color="primary",
        dark=true,
        className="mt-2 rounded")

    mechanism_input = dbc_row(className="mb-1") do
        dbc_label("Mechanism", width=4, html_for="mechanism"),
        dbc_col(className="", width=6) do 
            dcc_dropdown(
                id="mechanism",
                options=[
                    (label = "Classic", value = "Classic"),
                    (label = "Deterministic MLE", value="DetMLE"),
                    (label = "MLE", value = "MLE"),
                    (label = "Dumb", value = "Dumb"),
                    (label = "Bayesian", value = "Bayesian"),
                ],
                value="DetMLE",
            )
        end
    end

    num_agents_input = dbc_row(className="mb-1") do
        dbc_label("Number of agents", width=4, html_for="num_agents"),
        dbc_col(className="", width=6) do 
            dbc_input(type="number", id="num_agents", value=2, min=2, max=max_num_agents)
        end
    end

    num_rounds_input = dbc_row(className="mb-1") do
        dbc_label("Number of rounds", width=4, html_for="num_rounds"),
        dbc_col(className="", width=6) do 
            dbc_input(type="number", id="num_rounds", value=100, min=1, max=max_num_rounds)
        end
    end

    contest_config = dbc_card(body=true, className="my-2") do 
        html_h4("Contest", className="card-title"),
        html_p(className="card-text") do 
            "Set the number of agents and rounds, as well as the type of mechanism."
        end,
        dbc_form() do
            mechanism_input,
            num_agents_input,
            num_rounds_input
        end
    end

    function create_row(i)
        dbc_row(id=(type="parameter-row", index=i), className="mb-2") do
            dbc_col("$(i)"),
            dbc_col(dbc_input(id=(type="initial-efforts", index=i), type="number", value=round(0.01+0.25*rand(), digits=2), min=0, max=1, step=0.01, required=true)),
            dbc_col(dbc_input(id=(type="p-values", index=i), type="number", value=1, min=0.00, max=1.00, step=0.1, required=true)),
            dbc_col(dbc_input(id=(type="alpha-values", index=i), type="number", value=1, min=0.00, max=1.00, step=0.01, required=true)),
            dbc_col(dbc_input(id=(type="delta-values", index=i), type="number", value=0.05, min=0.00, max=1.00, step=0.01, required=true)),
            dbc_col(dbc_input(id=(type="window-values", index=i), type="number", value=50, min=0, required=true))
        end
    end

    header_row = dbc_row(id="parameter-headers") do 
        dbc_col(dbc_label("Agent")),
        dbc_col(dbc_label("Initial efforts")),
        dbc_col(dbc_label("p")),
        dbc_col(dbc_label("α")),
        dbc_col(dbc_label("δ")),
        dbc_col(dbc_label("window"))
    end

    agent_config = dbc_card(body=true, className="my-2") do 
        html_h4("Agent Parameters", className="card-title"),
        html_p(className="card-text") do 
        "Set all parameters for agents."
        end,
        header_row,
        [create_row(i) for i in 1:max_num_agents]...
    end

    graphs = dcc_loading(type="circle") do
        dbc_card(body=true, id="graphs", className="my-2") do
            html_h4("Outcomes"),
            dcc_graph(id="effort-graph", figure=plot(A)),
            dbc_row(className="g-2") do
                dbc_col(dcc_graph(id="utility-graph", figure=plot(A)), width=6),
                dbc_col(dcc_graph(id="nash-gap-graph", figure=plot(A)), width=6)
            end
        end
    end

    info = dbc_card(body=true, className="mb-2")do
        html_h4("Information"),
        html_p("Here's some background information about the Tullock contest dynamics implemented here."),
        html_p("[To be completed.]")
    end

    app.layout = dbc_container() do
        navbar,
        dbc_row(className="g-2") do
            dbc_col(width=5) do
                contest_config
            end,
            dbc_col(width=7) do
                agent_config
            end
        end,
        html_div(
            dbc_button(
                "Run Contest ▶️",
                id="compute-button",
                color="primary",
                className="mt-1 mb-1"
            ),
            className="text-left"
            ),
        dcc_store(id="contest_efforts"),
        dcc_store(id="contest_utilities"),
        dcc_store(id="nash_gaps"),
        graphs,
        info
        # html_hr()
    end

    # This callback ensures that only the parameters for the active agents are visible
    callback!(app,
            Output((type="parameter-row", index=ALL), "className"),
            Input("num_agents", "value")) do n
        v = vcat(["mb-1 d-flex" for i in 1:n], ["mb-1 d-none" for i in n+1:max_num_agents])
        return v
    end

    # This callback runs the contest and computes the graphs
    callback!(app,
            Output("effort-graph", "figure"),
            Output("utility-graph", "figure"),
            Output("nash-gap-graph", "figure"),
            Input("compute-button", "n_clicks"),
            State("mechanism", "value"),
            State("num_agents", "value"),
            State("num_rounds", "value"),
            State((type="initial-efforts", index=ALL), "value"),
            State((type="p-values", index=ALL), "value"),
            State((type="alpha-values", index=ALL), "value"),
            State((type="delta-values", index=ALL), "value"),
            State((type="window-values", index=ALL), "value")) do n_clicks, mechanism, n, T, x, p, α, δ, windows
        estimator = select_estimator(mechanism)
        contest = create_contest(estimator, n, T, x, p, α, δ, windows)
        run!(contest)
        effort_plt = create_plot(contest.efforts; title="Efforts over time")
        utility_plt = create_plot(contest.utilities; title="Utilies over time")
        nash_gaps = vec(sum(contest.nash_gaps, dims=1))
        nash_gap_plt = create_plot(nash_gaps; title="Nash gap over time")
        return effort_plt, utility_plt, nash_gap_plt
    end


    function create_contest(estimator, n, T, x, p, α, δ, windows)
        agents = [
            Agent(
                x -> x,  # linear costs
                t -> p[i],  # function p
                t -> max(1,t-windows[i]):t-1,  # window function
                t -> α[i],  # function α
                estimator,
                δ[i]  # function δ
                )
            for i in 1:n
        ]
        efforts = convert.(Float64, x[1:n])
        TullockContest(agents, efforts, T)
    end


    function select_estimator(mechanism::String)
        if mechanism == "Classic"; return classic_estimator; end
        if mechanism == "DetMLE"; return deterministic_max_likelihood_estimator; end
        if mechanism == "MLE"; return max_likelihood_estimator; end
        if mechanism == "Dumb"; return dumb_estimator; end
        # if mechanism == "Bayesian"; return bayesian_estimator; end
        throw("Mechanism not yet implemented!")
    end

    function create_plot(M::Matrix; title)
        n = size(M)[1]
        agentlabels = permutedims(["Agent $(i)" for i ∈ 1:n])
        plt = plot(
            M',
            Layout(
                height=250,
                title=title,
                plot_bgcolor="white",
                yaxis=attr(showline=true, linewidth=2, linecolor="gray", ticks="outside", tickwidth=2, tickcolor="gray"),
                xaxis=attr(showline=true, linewidth=2, linecolor="gray", ticks="outside", tickwidth=2, tickcolor="gray")
            )
        )
        restyle!(plt, 1:n, name=agentlabels)
        return plt    
    end


    function create_plot(V::Vector; title)
        n = length(V)
        plt = plot(
            V,
            Layout(
                height=250,
                title=title,
                plot_bgcolor="white",
                yaxis=attr(showline=true, linewidth=2, linecolor="gray", ticks="outside", tickwidth=2, tickcolor="gray"),
                xaxis=attr(showline=true, linewidth=2, linecolor="gray", ticks="outside", tickwidth=2, tickcolor="gray")
            )
        )
        return plt    
    end

    @info("Starting server...")
    run_server(app, host, port; debug=DEV_MODE)
end

end  # module