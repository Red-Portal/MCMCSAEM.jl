
include("common.jl")

using Distributed
using GLMNet
using DelimitedFiles

struct StudentTARD{F <: Real, Mat <: AbstractMatrix, Vec <: AbstractVector}
    α_noise::F
    β_noise::F
    X::Mat
    y::Vec
end

function LogDensityProblems.capabilities(::Type{<:StudentTARD})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::StudentTARD, z::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y, α_noise, β_noise = model
    d = size(X,2)

    α = z[1] 
    β = z[2:end] 
    γ = 1 ./ θ

    s = X*β .+ α
    ν = 2*α_noise
    σ = sqrt(β_noise/α_noise)
    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(TDist(ν), (yᵢ - sᵢ)/σ)
    end
    ℓp_β = logpdf(MvNormal(Zeros(d), γ), β)
    ℓp_α = logpdf(Normal(0, 10), α)
    ℓp_x + ℓp_β + ℓp_α
end

function MCMCSAEM.preconditioner(::StudentTARD, θ::AbstractVector)
    ϵ = eps(eltype(θ))
    Diagonal(vcat([1.0], @. 1 / (θ^2 + ϵ) + 1e-3))
end

function MCMCSAEM.sufficient_statistic(::StudentTARD, x::AbstractMatrix)
    mean(eachcol(x[2:end,:])) do xᵢ
        xᵢ.^2
    end
end

function MCMCSAEM.maximize_surrogate(::StudentTARD, S::AbstractVector)
    ϵ   = eps(eltype(S))
    EX² = S
    σ   = sqrt.(EX²) 
    @. 1 ./ (σ + ϵ) + ϵ
end

function load_dataset()
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

function predictive_loglikelihood(::StudentTARD, X, y, β_post, α_noise, β_noise)
    s = X*β_post
    ν = α_noise
    σ = sqrt(β_noise/α_noise)
    @tullio ℓp_y[i,j] := logpdf(TDist(ν), (y[i] - s[i,j])/σ)
    mean(logsumexp(ℓp_y, dims=2) .- log(size(β_post,2)))
end

function predictive_rmse(::StudentTARD, X, y, β_post)
    s      = X*β_post
    y_pred = mean(s, dims=2)[:,1]
    sqrt(mean(abs2, y_pred - y))
end

function run_problem(::Val{:studenttard}, dataset, mcmc_type, h, key=1, show_progress=true)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    ad   = ADTypes.AutoReverseDiff()

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y; ratio=0.8)

    d = size(X_train, 2)

    T_burn    = 1000
    T         = 20000
    γ₀        = 1e-0
    γ         = t -> γ₀/sqrt(t)
    m         = 1    # n_chains

    model = LogisticARD(X_train, y_train)
    θ₀    = fill(2.0, d)
    β     = rand(rng, MvNormal(Zeros(d), 1 ./ θ₀))
    α     = [0.0]
    x₀    = reshape(repeat(vcat(α, β), outer=m), (:,m))

    #y_glm_train = hcat(Int.(.!y_train), Int.(y_train))
    #lasso_model = glmnet(X_train, y_glm_train, Binomial(); intercept=true)
    #lasso_sel   = lasso_model.betas[:,end] .> 0

    #θ_hist = zeros(length(θ₀), T)
    V_hist = zeros(T)
    function callback!(t, x, θ, stats)
        V_hist[t] = stats.loglike
        nothing
    end

    θ, x = MCMCSAEM.mcmcsaem(rng, model, x₀, θ₀, T, T_burn, γ, h;
                             ad, callback!, show_progress, mcmc_type)
    #θ = mean(θ_hist, dims=2)[:,1]
    #Plots.plot!(1 ./ θ) |> display
    #Plots.plot!(-abs.(lasso_model.betas[:,end])) |> display
    #Plots.plot(abs.(x[6001:end])) |> display
    #Plots.plot!(log.(mean(θ_hist, dims=2)[:,1])) |> display
    Plots.plot(V_hist) |> display

    #θ = @. abs(lasso_model.betas[:,end]) + 1e-2
    #x = x₀

    #idx_m = argmin(θ)
    #idx_p = argmax(θ)
    #Plots.plot(log.(θ_hist[[idx_p, idx_m],:]')) |> display
    #Plots.plot(V_hist) |> display

    β_post = MCMCSAEM.mcmc(rng, model, θ, x[:,end], 1e-3, 5000; ad, show_progress)
    X_test = hcat(ones(size(X_test,1)), X_test)

    lpd = predictive_loglikelihood(model, X_test, y_test, β_post)
    acc = predictive_rmse(         model, X_test, y_test, β_post)

    GC.gc()

    DataFrame(lpd=lpd, acc=acc)
end

function main(::Val{:studenttard}, mcmc_type)
    #@everywhere run(`taskset -pc $(myid() - 1) $(getpid())`)

    n_trials = 64
    datasets = [
        (dataset = :mushroom,),
        (dataset = :sonar,),
        (dataset = :caravan,),
        (dataset = :phishing,),
    ]
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-5, 0, length=11) ]

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

    h5open(datadir("exp_pro", "logisticard_$(mcmc_type).h5"), "w") do h5
        for dataset ∈ [:mushroom, :sonar, :caravan, :phishing]
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
