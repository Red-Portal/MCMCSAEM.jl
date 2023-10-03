
include("common.jl")

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

function MCMCSAEM.project(::LogisticARD, θ::AbstractVector)
    @. clamp(θ, 1e-3, Inf)
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
    X = X .- mean(X, dims=1)
    X = X ./ std(X, dims=1)
    y = y .== 1.0
    X, y
end

function load_dataset(::Val{:prostate})
    mat  = MAT.matread(datadir("dataset", "prostate.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    X = X .- mean(X, dims=1)
    X = X ./ std(X, dims=1)
    y = y .== 2.0
    X, y
end

function run(::Val{:logisticard}, dataset, h, key=1, show_progress=true)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    ad   = ADTypes.AutoReverseDiff()

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y)

    d = size(X_train, 2)

    T_burn    = 500
    T         = 1000
    γ₀        = 1e-0
    γ         = t -> γ₀ / t
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

    θ, x = MCMCSAEM.mcmcsaem(rng, model, x₀, θ₀, T, T_burn, γ, h; ad, callback!, show_progress)
    #Plots.plot!(1 ./ θ) |> display
    #Plots.plot(V_hist)

    # model_sel = LogisticARD(X_train[:,select_idx], y_train)
    # θ_sel     = θ[select_idx]
    # x_sel     = x[vcat([1], select_idx .+ 1),:]
    # β_post    = MCMCSAEM.mcmc(rng, model_sel, θ_sel, x_sel, 1e-4, 2000; ad)
    # β_post    = β_post[:,1000:10:end]
    # X_test    = hcat(ones(size(X_test,1)), X_test[:,select_idx])

    θ = mean(θ_hist, dims=2)[:,1]

    β_post = MCMCSAEM.mcmc(rng, model, θ, x, 1e-3, 2000; ad, show_progress)
    X_test = hcat(ones(size(X_test,1)), X_test)

    logits = X_test*β_post
    p_test = mean(logistic.(logits), dims=2)[:,1]

    acc  = mean((p_test .> 0.5) .== y_test)
    mlpd = mapreduce((pᵢ, yᵢ) -> logpdf(Bernoulli(pᵢ), yᵢ), +, p_test, y_test) / length(y_test)

    DataFrame(acc=acc, mlpd=mlpd)
end

function main(::Val{:logisticard})
    n_trials = 32
    datasets = [
        (dataset = :colon,),
        (dataset = :prostate,),
        (dataset = :leukemia,)
    ]
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-6, -2, length=11) ]

    configs = Iterators.product(datasets, stepsizes) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    data = @showprogress mapreduce(vcat, configs) do config
        SimpleUnPack.@unpack stepsize, dataset = config
        dfs = @showprogress pmap(1:n_trials) do key
            run(Val(:logisticard), Val(dataset), stepsize, key, false)
        end
        df = vcat(dfs...)
        for (k, v) ∈ pairs(config)
            df[:,k] .= v
        end
        GC.gc()
        df
    end

    JLD2.save(datadir("exp_pro", "logisticard_accuracy.jld2"), "data", data)

    function run_bootstrap(data′)
        boot = bootstrap(mean, data′, BalancedSampling(1024))
        confint(boot, PercentileConfInt(0.8)) |> only
    end

    h5open(datadir("exp_pro", "logisticard_accuracy.h5"), "w") do h5
        for dataset ∈ [:colon, :leukemia, :prostate]
            data′ = data[data[:,:dataset] .== dataset,:]
            data′′ = @chain groupby(data′, :stepsize) begin
                @combine(:acc_ci    = run_bootstrap(:acc),
                         :mlpd_ci   = run_bootstrap(:mlpd))
            end
            h  = data′′[:,:stepsize]
            
            acc      = data′′[:,:acc_ci]
            acc_mean = [accᵢ[1] for accᵢ ∈ acc]
            acc_p    = [abs(accᵢ[2] - accᵢ[1]) for accᵢ ∈ acc]
            acc_m    = [abs(accᵢ[3] - accᵢ[1]) for accᵢ ∈ acc]
            
            mlpd      = data′′[:,:mlpd_ci]
            mlpd_mean = [mlpdᵢ[1] for mlpdᵢ ∈ mlpd]
            mlpd_p    = [abs(mlpdᵢ[2] - mlpdᵢ[1]) for mlpdᵢ ∈ acc]
            mlpd_m    = [abs(mlpdᵢ[3] - mlpdᵢ[1]) for mlpdᵢ ∈ acc]

            write(h5, "h_$(dataset)",    h)
            write(h5, "acc_$(dataset)",  hcat( acc_mean,  acc_p,  acc_m)' |> Array)
            write(h5, "mlpd_$(dataset)", hcat(mlpd_mean, mlpd_p, mlpd_m)' |> Array)
        end
    end
    data
end
