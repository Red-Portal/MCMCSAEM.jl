
include("common.jl")

using GLMNet

struct LogisticARD{Mat <: AbstractMatrix, Vec <: AbstractVector}
    X::Mat
    y::Vec
end

function LogDensityProblems.capabilities(::Type{<:LogisticARD})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::LogisticARD, β::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y = model
    d = size(X,2)
    σ = 1 ./ θ
    s = X*β[2:end] .+ β[1]

    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(BernoulliLogit(sᵢ), yᵢ)
    end
    ℓp_β = logpdf(MvNormal(Zeros(d), σ), β[2:end])
    ℓp_α = logpdf(Normal(0, 10), β[1])
    ℓp_x + ℓp_β + ℓp_α
end

function MCMCSAEM.preconditioner(::LogisticARD, θ::AbstractVector)
    Diagonal(vcat([0.1], (@. 1/θ^2 + 1e-7)))
end

function MCMCSAEM.sufficient_statistic(::LogisticARD, x::AbstractMatrix)
    mean(eachcol(x[2:end,:])) do xᵢ
        xᵢ.^2
    end
end

function MCMCSAEM.maximize_surrogate(::LogisticARD, S::AbstractVector)
    ϵ   = eps(eltype(S))
    EX² = S
    σ²  = EX²
    @. 1.0 / (sqrt(σ²) + ϵ)
end

function load_dataset(::Val{:colon})
    mat  = MAT.matread(datadir("dataset", "colon.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    y    = y .== 1.0
    X, y
end

function load_dataset(::Val{:leukemia})
    mat  = MAT.matread(datadir("dataset", "leukemia.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    y    = y .== 1.0
    X, y
end

function load_dataset(::Val{:prostate})
    mat  = MAT.matread(datadir("dataset", "prostate.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    y    = y .== 2.0
    X, y
end

function lasso(dataset, key=1)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test = prepare_dataset(rng, X, y; ratio=0.8)

    y_glm_train = hcat(Int.(.!y_train), Int.(y_train))
    glm         = glmnetcv(X_train, y_glm_train, Binomial();
                           intercept=true, grouped=false) 
    
    p_test = predict(glm, X_test, outtype=:prob)
    acc    = mean((p_test .> 0.5) .== y_test)

    mlpd   = mean(map((pᵢ, yᵢ) -> logpdf(Bernoulli(pᵢ), yᵢ), p_test, y_test))
    acc, mlpd
end

function run_lasso_dataset(dataset)
    n_trials = 32
    mlpd     = map(1:n_trials) do key
        lasso(dataset, key)[2]
    end
    ci = run_bootstrap(mlpd)
    @info("", dataset, mean = ci[1], Δci = (abs(ci[2] - ci[1]), abs(ci[3] - ci[1])))
end

function run_lasso_all()
    run_lasso_dataset(Val(:colon))
    run_lasso_dataset(Val(:prostate))
    run_lasso_dataset(Val(:leukemia))
end

function predictive_loglikelihood(::LogisticARD, X, y, β_post)
    logits = X*β_post
    @tullio ℓp_y[i,j] := logpdf(BernoulliLogit(logits[i,j]), y[i])
    mean(logsumexp(ℓp_y, dims=2) .- log(size(β_post,2)))
end

function run_problem(::Val{:logisticard}, dataset, mcmc_type, h, key=1, show_progress=true)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    ad   = ADTypes.AutoReverseDiff()

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y; ratio=0.8)

    d = size(X_train, 2)

    T_burn    = 1000
    T         = 50000
    γ₀        = 1e-0
    γ         = t -> γ₀/sqrt(t)
    m         = 1    # n_chains

    model = LogisticARD(X_train, y_train)
    σ₀    = 0.1
    θ₀    = fill(1/σ₀, d)
    x₀    = σ₀*randn(rng, d+1, m)

    θ_hist = zeros(length(θ₀), T)
    V_hist = zeros(T)
    function callback!(t, x, θ, stats)
        θ_hist[:,t] = θ
        V_hist[t]   = stats.loglike
        nothing
    end

    θ, x = MCMCSAEM.mcmcsaem(rng, model, x₀, θ₀, T, T_burn, γ, h;
                             ad, callback!, show_progress, mcmc_type)
    Plots.plot!(1 ./ θ) |> display
    #Plots.plot!(log.(mean(θ_hist, dims=2)[:,1])) |> display
    #Plots.plot(V_hist) |> display
    #throw()

    β_post = MCMCSAEM.mcmc(rng, model, θ, x, 1e-3, 5000; ad, show_progress)
    X_test = hcat(ones(size(X_test,1)), X_test)

    # p_test_samples = logistic.(X_test*β_post)
    # p_test         = map(eachrow(p_test_samples)) do p_testᵢ
    #     y_predᵢ = vcat((@. rand(rng, Bernoulli(p_testᵢ), 100))...)
    #     mean(y_predᵢ)
    # end
    # acc = mean((p_test .> 0.5) .== y_test)
    # lpd = mapreduce((pᵢ, yᵢ) -> logpdf(Bernoulli(pᵢ), yᵢ), +, p_test, y_test) / length(y_test)
    lpd = predictive_loglikelihood(model, X_test, y_test, β_post)

    GC.gc()

    DataFrame(lpd=lpd)
end

function main(::Val{:logisticard})
    n_trials = 32
    datasets = [
        (dataset = :colon,),
        (dataset = :prostate,),
        (dataset = :leukemia,)
    ]
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-5, -2, length=11) ]

    configs = Iterators.product(datasets, stepsizes) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    data = @showprogress mapreduce(vcat, configs) do config
        SimpleUnPack.@unpack stepsize, dataset = config
        dfs = @showprogress pmap(1:n_trials) do key
            run_problem(Val(:logisticard), Val(dataset), stepsize, key, false)
        end
        df = vcat(dfs...)
        for (k, v) ∈ pairs(config)
            df[:,k] .= v
        end
        df
    end

    JLD2.save(datadir("exp_pro", "logisticard_accuracy.jld2"), "data", data)

    h5open(datadir("exp_pro", "logisticard_accuracy.h5"), "w") do h5
        for dataset ∈ [:colon, :leukemia, :prostate]
            data′ = data[data[:,:dataset] .== dataset,:]
            data′′ = @chain groupby(data′, :stepsize) begin
                @combine(:lpd_ci   = run_bootstrap(:lpd))
            end
            h  = data′′[:,:stepsize]
            
            lpd      = data′′[:,:lpd_ci]
            lpd_mean = [lpdᵢ[1] for lpdᵢ ∈ lpd]
            lpd_p    = [abs(lpdᵢ[2] - lpdᵢ[1]) for lpdᵢ ∈ lpd]
            lpd_m    = [abs(lpdᵢ[3] - lpdᵢ[1]) for lpdᵢ ∈ lpd]

            write(h5, "h_$(dataset)",   h)
            write(h5, "lpd_$(dataset)", hcat(lpd_mean, lpd_p, lpd_m)' |> Array)
        end
    end
    data
end
