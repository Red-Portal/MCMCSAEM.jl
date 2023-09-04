
using DrWatson

using Zygote
using ADTypes
using Optimisers

using Accessors
using DelimitedFiles
using Distributions
using FillArrays
using Flux
using LinearAlgebra
using LogDensityProblems
using MAT
using Optimisers
using Plots
using Random
using Random123
using SimpleUnPack

using AdvancedVI
using MCMCSAEM

include("dataset.jl")

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    #CUDA.allowscalar(false)

    use_cuda = true

    X, y   = load_dataset(Val(:wine))
    X_train, y_train, X_test, y_test = prepare_dataset(rng, X, y)

    μ_y_train    = mean(y_train)
    σ_y_train    = std(y_train)
    y_train_cent = @. (y_train - μ_y_train) / σ_y_train
    prob         = BNN(X_train, y_train_cent, 50)

    batchsize    = 30
    n_samples    = 10

    advi       = ADVI(prob, n_samples)
    advidoubly = Subsampling(advi, batchsize, 1:length(y_train))

    d     = LogDensityProblems.dimension(prob)

    @info("", d = d)

    #q     = StructuredLocationScale(prob; use_cuda)

    q     = VIMeanFieldGaussian(zeros(Float32, d), Diagonal(ones(Float32, d)))
    #q     = VIMeanFieldGaussian(CUDA.zeros(Float32, d), Diagonal(.1f0*CUDA.ones(Float32, d)))
    λ, re = Optimisers.destructure(q)

    callback!(; λ, args...) = begin
        q = re(λ)

        bnn_locat = prob.restruct(q.location)
        bnn_scale = prob.restruct(diag(q.scale))

        μ_η_γ⁻¹ = only(bnn_locat.η_γ⁻¹)
        σ_η_γ⁻¹ = only(bnn_scale.η_γ⁻¹)

        q_pred = MCMCSAEM.bnn_adf_predict(
            bnn_locat.W₁, bnn_scale.W₁,
            bnn_locat.W₂, bnn_scale.W₂,
            μ_η_γ⁻¹,      σ_η_γ⁻¹,
            μ_y_train,    σ_y_train,
            vcat(X_test, ones(1, size(X_test,2)))
        )
        lpd  = logpdf(q_pred, y_test) / length(y_test)
        rmse = sqrt(Flux.Losses.mse(mean(q_pred), y_test, agg=mean))
        if any(@. isnan(λ) | isinf(λ))
            throw(ErrorException("NaN detected"))
        end
        (lpd=lpd, rmse=rmse,)
    end

    n_max_iter = 10^4
    q, stats, _ = optimize(
        advidoubly,
        q,
        n_max_iter;
        callback! = callback!,
        rng       = rng,
        adbackend = ADTypes.AutoZygote(),
        optimizer = Optimisers.Adam(1f-3)
    )
    elbo = [stat.lpd for stat ∈ stats]
    plot(elbo, ylims=quantile(elbo, (0.1, 1.))) |> display
    q
end
