
include("common.jl")

using Distributed
using GLMNet
using DelimitedFiles

struct LogisticARD{Mat <: AbstractMatrix, Vec <: AbstractVector}
    X::Mat
    y::Vec
end

function LogDensityProblems.capabilities(::Type{<:LogisticARD})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::LogisticARD, z::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y = model
    d = size(X,2)

    α = z[1] 
    β = z[2:end] 
    γ = 1 ./ θ

    s = X*β .+ α
    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(BernoulliLogit(sᵢ), yᵢ)
    end
    ℓp_β = logpdf(MvNormal(Zeros(d), γ), β)
    ℓp_α = logpdf(Normal(0, 10), α)
    ℓp_x + ℓp_β + ℓp_α
end

function MCMCSAEM.preconditioner(::LogisticARD, θ::AbstractVector)
    ϵ = eps(eltype(θ))
    Diagonal(vcat([1.0], @. 1 / (θ^2 + 0.01) + ϵ))
end

function MCMCSAEM.sufficient_statistic(::LogisticARD, x::AbstractMatrix)
    mean(eachcol(x[2:end,:])) do xᵢ
        xᵢ.^2
    end
end

function MCMCSAEM.maximize_surrogate(::LogisticARD, S::AbstractVector)
    ϵ   = sqrt(eps(eltype(S)))
    EX² = S
    σ   = sqrt.(EX²) 
    @. 1 ./ (σ + ϵ) + ϵ
end

function onehot(
    df::AbstractDataFrame,
    col,
    cate = sort(unique(df[!, col]));
    outname = :ohe_
)
    outnames = Symbol.(outname, cate)
    transform(df, @. col => ByRow(isequal(cate)) .=> outnames)
end

function load_dataset(::Val{:german})
    data = readdlm(datadir("dataset", "german.data-numeric"))
    y    = data[:,end] .== 2.0
    X    = data[:,1:end-1]

    df = DataFrame(X, :auto)
    for col ∈ [:x1, :x3, :x4, :x9, :x10, :x12, :x14, :x15, :x17]
        df = onehot(df, col, outname = col)
        df = select(df, Not(col))
    end
    X = Array(df)
    X, y
end

function load_dataset(::Val{:australian})
    data = readdlm(datadir("dataset", "australian.dat"), ' ')
    y    = Vector(data[:,end] .== 1.0)
    X    = Matrix{Float64}(data[:,1:end-1])

    numeric_idx = [2,3,5,7,10,13,14]
    X[:,numeric_idx] .-= mean(X[:,numeric_idx], dims=1)
    X[:,numeric_idx] ./= std(X[:,numeric_idx], dims=1)

    categorical_idx = setdiff(1:14, numeric_idx)
    df = DataFrame(X, :auto)
    for col ∈ [Symbol("x$(idx)") for idx ∈ categorical_idx]
        df = onehot(df, col, outname = col)
        df = select(df, Not(col))
    end
    X = Array(df)
    X, y
end

function load_dataset(::Val{:sonar})
    data = readdlm(datadir("dataset", "sonar data.csv"), ',')
    y    = Vector(data[:,end] .== "M")
    X    = Matrix{Float64}(data[:,1:end-1])

    X .-= mean(X, dims=1)
    X ./= std(X, dims=1)

    X, y
end

function load_dataset(::Val{:breast})
    data = readdlm(datadir("dataset", "wdbc.data"), ',')
    y    = Vector(data[:,2] .== "M")
    X    = Matrix{Float64}(data[:,3:end])

    X .-= mean(X, dims=1)
    X ./= std(X, dims=1)

    X, y
end

function load_dataset(::Val{:caravan})
    data, _ = readdlm(datadir("dataset", "caravan.csv"), ',', header=true)
    y       = Vector(data[:,end] .== 1.0)
    X       = Matrix{Float64}(data[:,2:end-1])

    df = DataFrame(X, :auto)
    for col ∈ [Symbol("x$(idx)") for idx ∈ 1:size(X,2)]
        df = onehot(df, col, outname = col)
        df = select(df, Not(col))
    end
    X = Array{Int}(df)

    X, y
end

function load_dataset(::Val{:phishing})
    data, _ = readdlm(datadir("dataset", "phishing.csv"), ',', header=true)
    y       = Vector(data[:,1] .> 0)
    X       = Matrix{Float64}(data[:,2:end])
    X, y
end

function lasso(dataset, key=1, show_result=false)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test = prepare_dataset(rng, X, y; ratio=0.8)

    y_glm_train = hcat(Int.(.!y_train), Int.(y_train))
    glm         = glmnetcv(X_train, y_glm_train, Binomial();
                           intercept=true, grouped=false) 

    if show_result
        glmnet(X_train, y_glm_train, Binomial(); intercept=true) |> display
    end

    p_test = predict(glm, X_test, outtype=:prob)
    p_test = @. clamp(p_test, 0, 1)
    acc    = mean((p_test .> 0.5) .== y_test)

    mlpd   = mean(map((pᵢ, yᵢ) -> logpdf(Bernoulli(pᵢ), yᵢ), p_test, y_test))
    acc, mlpd
end

function run_lasso_dataset(dataset)
    n_trials = 32
    data     = map(1:n_trials) do key
        lasso(dataset, key)
    end
    lpd    = [datum[2] for datum in data]
    acc    = [datum[1] for datum in data]
    lpd_ci = run_bootstrap(lpd)
    acc_ci = run_bootstrap(acc)
    @info("",
          dataset,
          lpd_mean = lpd_ci[1],
          lpd_Δci  = (abs(lpd_ci[2] - lpd_ci[1]), abs(lpd_ci[3] - lpd_ci[1])),
          acc_mean = acc_ci[1],
          acc_Δci  = (abs(acc_ci[2] - acc_ci[1]), abs(acc_ci[3] - acc_ci[1])))
end

function run_lasso_all()
    run_lasso_dataset(Val(:german))
    run_lasso_dataset(Val(:sonar))
    run_lasso_dataset(Val(:phishing))
    run_lasso_dataset(Val(:caravan))
end

function predictive_loglikelihood(::LogisticARD, X, y, β_post)
    logits = X*β_post
    @tullio ℓp_y[i,j] := logpdf(BernoulliLogit(logits[i,j]), y[i])
    mean(logsumexp(ℓp_y, dims=2) .- log(size(β_post,2)))
end

function predictive_accuracy(::LogisticARD, X, y, β_post)
    logits = X*β_post
    p_y    = mean(logistic.(logits), dims=2)[:,1]
    mean((p_y .> 0.5) .== y)
end

function run_problem(::Val{:logisticard}, dataset, mcmc_type, h, key=1, show_progress=true)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    ad   = ADTypes.AutoReverseDiff()

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y; ratio=0.8)

    d = size(X_train, 2)

    T_burn    = 100
    T         = 2000
    γ₀        = 1e-0
    γ         = t -> γ₀/sqrt(t)
    m         = 1    # n_chains

    n_inner_mcmc = 4

    model = LogisticARD(X_train, y_train)
    θ₀    = fill(1.0, d)
    β     = rand(rng, MvNormal(Zeros(d), 1 ./ θ₀))
    α     = [0.0]
    x₀    = reshape(repeat(vcat(α, β), outer=m), (:,m))

    θ, x, _ = MCMCSAEM.mcmcsaem(
        rng, model, x₀, θ₀, T, T_burn, γ, h;
        ad, show_progress, mcmc_type,
        n_inner_mcmc=n_inner_mcmc
    )

    #stats_loglike = filter(Base.Fix2(haskey, :loglike), stats)
    #Plots.plot([stat.loglike for stat in stats_loglike]) |> display

    β_post = MCMCSAEM.mcmc(rng, model, θ, x[:,1], 1e-3, 4000; ad, show_progress)
    X_test = hcat(ones(size(X_test,1)), X_test)

    lpd = predictive_loglikelihood(model, X_test, y_test, β_post)
    acc = predictive_accuracy(     model, X_test, y_test, β_post)

    GC.gc()

    lpd = isfinite(lpd) ? lpd : -1000.0
    acc = isfinite(acc) ? acc : 0.0

    DataFrame(lpd=lpd, acc=acc)
end

function main(::Val{:logisticard}, mcmc_type)
    n_trials = 32
    datasets = [
        (dataset = :german,),
        (dataset = :phishing,),
        (dataset = :caravan,),
    ]
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-5, -2, length=16) ]

    configs = Iterators.product(datasets, stepsizes) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    data = @showprogress mapreduce(vcat, configs) do config
        SimpleUnPack.@unpack stepsize, dataset = config
        dfs = @showprogress pmap(1:n_trials) do key
            run_problem(Val(:logisticard), Val(dataset), mcmc_type, stepsize, key, false)
        end
        df = vcat(dfs...)
        for (k, v) ∈ pairs(config)
            df[:,k] .= v
        end
        df
    end

    JLD2.save(datadir("exp_pro", "logisticard_$(mcmc_type).jld2"), "data", data)
    data = JLD2.load(datadir("exp_pro", "logisticard_$(mcmc_type).jld2"), "data")

    h5open(datadir("exp_pro", "logisticard_$(mcmc_type).h5"), "w") do h5
        for dataset ∈ [:german, :phishing, :caravan]
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
