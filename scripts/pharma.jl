
include("common.jl")

struct PharmaNLME{
    V   <: AbstractVector,
    I   <: AbstractVector{<:Integer},
    Ind <: AbstractRange,
    T   <: Real
}
    t        ::V
    dosage   ::V
    weight   ::V

    y        ::V
    subject  ::I

    n_subject::Int

    ka_idx ::Ind
    vol_idx::Ind
    clr_idx::Ind

    temp   ::T
end

function PharmaNLME(
    dosage ::AbstractVector,
    weight ::AbstractVector,
    t      ::AbstractVector,
    y      ::AbstractVector,
    subject::AbstractVector{<:Integer}
)
    @assert length(t)       == length(y)
    @assert length(subject) == length(t)
    @assert length(weight)  == length(dosage)

    n_subject = length(unique(subject))
    ka_idx    = 1:n_subject
    vol_idx   = n_subject + 1:2*n_subject
    clr_idx   = 2*n_subject + 1:3*n_subject

    #vol_idx   = 1:n_subject
    #clr_idx   = n_subject + 1:2*n_subject
    #ka_idx    = 2*n_subject + 1:3*n_subject

    PharmaNLME(
        t, dosage, weight, y, subject, n_subject,
        ka_idx, vol_idx, clr_idx, 1.0
    )
end

function LogDensityProblems.dimension(model::PharmaNLME)
    SimpleUnPack.@unpack n_subject, y = model
    3*n_subject
end

function LogDensityProblems.capabilities(::Type{<:PharmaNLME})
    LogDensityProblems.LogDensityOrder{0}()
end

function model1cptmt(ka::Real, clr::Real, dos::Real, vol::Real, t::Real)
    dos*ka/(vol*ka - clr)*(exp(-clr/vol*t) - exp(-ka*t))
end

function LogDensityProblems.logdensity(
    model::PharmaNLME, z::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack weight, dosage, t, y, subject, n_subject, vol_idx, clr_idx, ka_idx, temp = model

    ℓvol = z[vol_idx]
    ℓclr = z[clr_idx]
    ℓka  = z[ka_idx]

    ℓμ_ka  = θ[1]
    ℓμ_vol = θ[2]
    ℓμ_clr = θ[3]

    σ_ka  = θ[4]
    σ_vol = θ[5]
    σ_clr = θ[6]
    a     = θ[7]

    vol = exp.(ℓvol)
    ka  = exp.(ℓka)
    clr = exp.(ℓclr)

    ℓp_ℓvol = logpdf(MvNormal(Fill(ℓμ_vol, n_subject), σ_vol), ℓvol)
    ℓp_ℓclr = logpdf(MvNormal(Fill(ℓμ_clr, n_subject), σ_clr), ℓclr)
    ℓp_ℓka  = logpdf(MvNormal(Fill(ℓμ_ka,  n_subject), σ_ka ), ℓka)

    dos_vec = dosage[subject]
    vol_vec = vol[   subject]
    clr_vec = clr[   subject]
    ka_vec  = ka[    subject]

    μ    = @. model1cptmt(ka_vec, clr_vec, dos_vec, vol_vec, t)
    ℓp_y = logpdf(MvNormal(μ, a), y)

    temp*ℓp_y + ℓp_ℓvol + ℓp_ℓclr + ℓp_ℓka
end

function MCMCSAEM.sufficient_statistic(
    model::PharmaNLME,
    x    ::AbstractMatrix,
)
    SimpleUnPack.@unpack t, y, dosage, subject, weight, vol_idx, clr_idx, ka_idx = model
    mean(eachcol(x)) do xi
        ℓka  = xi[ka_idx]
        ℓvol = xi[vol_idx]
        ℓclr = xi[clr_idx]

        vol   = exp.(ℓvol)
        ka    = exp.(ℓka)
        clr   = exp.(ℓclr)

        clr_vec = clr[   subject]
        dos_vec = dosage[subject]
        vol_vec = vol[   subject]
        ka_vec  = ka[    subject]
        μ       = @. model1cptmt(ka_vec, clr_vec, dos_vec, vol_vec, t)
        ϵ       = y - μ

        Ex_ka  = mean(ℓka)
        Ex_vol = mean(ℓvol)
        Ex_clr = mean(ℓclr)

        Ex2_ka  = mean(ℓka.^2)
        Ex2_vol = mean(ℓvol.^2)
        Ex2_clr = mean(ℓclr.^2)
        Eϵ2     = mean(ϵ.^2)

        [Ex_ka, Ex2_ka, Ex_vol, Ex2_vol, Ex_clr, Ex2_clr, Eϵ2]
    end
end

function MCMCSAEM.preconditioner(model::PharmaNLME, θ::AbstractVector)
        I
end

function MCMCSAEM.maximize_surrogate(model::PharmaNLME, S::AbstractVector)
    SimpleUnPack.@unpack weight, dosage, t, n_subject = model

    Ex_ka, Ex2_ka   = S[1], S[2]
    Ex_vol, Ex2_vol = S[3], S[4]
    Ex_clr, Ex2_clr = S[5], S[6]
    Eϵ2             = S[7]

    σ_ℓka  = sqrt(Ex2_ka  - Ex_ka.^2)
    σ_ℓvol = sqrt(Ex2_vol - Ex_vol.^2)
    σ_ℓclr = sqrt(Ex2_clr - Ex_clr.^2)
    a      = sqrt(Eϵ2)

    [ Ex_ka, Ex_vol, Ex_clr, σ_ℓka, σ_ℓvol, σ_ℓclr, a ]
end

function load_dataset(rng::Random.AbstractRNG, ::Val{:pharma})
    data   = readdlm(datadir("theophylline_saemix.csv"), ',', Any, '\n')
    data   = DataFrame(identity.(data), ["Subject", "Dose", "Time", "Conc", "Wt", "Sex"])
    groups = groupby(data, :Subject)

    t       = [Array{Float64}(group.Time) for group in groups]
    y       = [Array{Float64}(group.Conc) for group in groups]
    dosage  = [Float64(first(group.Dose)) for group in groups]
    weight  = [Float64(first(group.Wt))   for group in groups]
    subject = collect(1:length(groups))
    t, dosage, y, weight, subject
end

function sample_prior(
    rng      ::Random.AbstractRNG,
    model    ::PharmaNLME,
    θ        ::AbstractVector,
    n_samples::Int
)
    n_subject = model.n_subject
    μ_z       = repeat(θ[1:3], inner=n_subject)
    σ_z       = repeat(θ[4:6], inner=n_subject)
    rand(rng, MvNormal(μ_z, σ_z), n_samples)
end

function logpdf_prior(
    model    ::PharmaNLME,
    θ        ::AbstractVector,
    z        ::AbstractVector
)
    n_subject = model.n_subject
    μ_z       = repeat(θ[1:3], inner=n_subject)
    σ_z       = repeat(θ[4:6], inner=n_subject)
    logpdf(MvNormal(μ_z, σ_z), z)
end

function run_problem(::Val{:pharma}, mcmc_type, h, key = 1, show_progress=true)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    #ad = ADTypes.AutoZygote()
    ad = ADTypes.AutoForwardDiff()

    t, dosage, y, weight, subject = load_dataset(rng, Val(:pharma))

    n_samples  = length(first(t))
    n_subjects = length(t)
    n_train    = floor(Int, 0.8*n_subjects)
    n_test     = n_subjects - n_train
    train_idx  = sample(rng, 1:n_subjects, n_train; replace=false)
    test_idx   = setdiff(collect(1:n_subjects), train_idx)

    t_test = vcat(t[test_idx]...)
    y_test = vcat(y[test_idx]...)
    subject_test = repeat(1:n_test, inner=length(first(t)))

    t_train       = vcat(t[train_idx]...)
    y_train       = vcat(y[train_idx]...)
    subject_train = repeat(1:n_train, inner=length(first(t)))

    model = PharmaNLME(
        dosage[train_idx], weight[train_idx], t_train, y_train, subject_train
    )


    T_burn    = 100
    T         = 1000
    γ₀        = 1e-0
    γ         = t -> γ₀/sqrt(t)

    n_inner_mcmc = 4

    #θ₀ = [log(1.),log(20),log(0.5),1,1,1,1]
    θ₀ = [0.,0.,0.,.1,.1,.1,2.0]
    x₀ = sample_prior(rng, model, θ₀, 1)

    function callback!(t, x, θ, stat)
        μ_ℓka  = θ[1]
        μ_ℓvol = θ[2]
        μ_ℓclr = θ[3]
        a      = θ[7]

        stat = (
            μ_ka       = μ_ℓka,
            μ_vol      = μ_ℓvol,
            μ_clr      = μ_ℓclr, 
            σ_ka       = θ[4],
            σ_vol      = θ[5],
            σ_clr      = θ[6],
            a          = a
         )

        if mod(t, 100) == 0 
            vol = exp(μ_ℓvol)
            ka  = exp(μ_ℓka)
            clr = exp(μ_ℓclr)

            dos_test = repeat(dosage[test_idx], inner=n_samples)
            μ_test   = @. model1cptmt(ka, clr, dos_test, vol, t_test)

            dos_train = repeat(dosage[train_idx], inner=n_samples)
            μ_train   = @. model1cptmt(ka, clr, dos_train, vol, t_train)

            merge(stat, (rmse_train = sqrt(mean(abs2, μ_train - y_train)),
                         rmse_test  = sqrt(mean(abs2, μ_test  - y_test)),))
        else
            stat
        end
    end

    θ, x, stats = MCMCSAEM.mcmcsaem(
        rng, model, x₀, θ₀, T, T_burn, γ, h;
        ad, callback! = callback!,
        show_progress = show_progress,
        mcmc_type,
        n_inner_mcmc  = n_inner_mcmc
    )
    stats_rmse = filter(Base.Fix2(haskey, :rmse_train), stats)
    if show_progress
        Plots.plot( [stat.rmse_train for stat in stats_rmse]) |> display
        Plots.plot!([stat.rmse_test  for stat in stats_rmse]) |> display
    end
    rmse = last([stat.rmse_test  for stat in stats_rmse])
    rmse = isnan(rmse) ? 10 : rmse
    
    test_model = PharmaNLME(
        dosage[test_idx], weight[test_idx], t_test, y_test, subject_test
    )

    n_subject = test_model.n_subject
    μ_z       = repeat(θ[1:3], inner=n_subject)
    σ_z       = repeat(θ[4:6], inner=n_subject)
    q0        = MvNormal(μ_z, σ_z)
    lml       = MCMCSAEM.ais(
        rng, test_model, θ, 1e-3, q0, range(0.,1.; length=1000).^2,
        100; ad, mcmc_type = :mala, show_progress = show_progress
    )
    lml  = isfinite(lml) ? lml : nextfloat(typemin(Float32))
    lpd  = lml/test_model.n_subject
    DataFrame(rmse=rmse, lpd=lpd, lml=lml)
end

function main(mcmc_type)
    n_trials  = 32
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-4, 0., length=17) ]
    configs   = stepsizes
        
    data = @showprogress mapreduce(vcat, configs) do config
        SimpleUnPack.@unpack stepsize = config
        dfs = @showprogress pmap(1:n_trials) do key
            run_problem(Val(:pharma), mcmc_type, stepsize, key, false)
        end
        df = vcat(dfs...)
        for (k, v) ∈ pairs(config)
            df[:,k] .= v
        end
        df
    end
        
    JLD2.save(datadir("exp_pro", "pharma_$(mcmc_type).jld2"), "data", data)
    data = JLD2.load(datadir("exp_pro", "pharma_$(mcmc_type).jld2"), "data")

    h5open(datadir("exp_pro", "pharma_$(mcmc_type).h5"), "w") do h5
        data′ = @chain groupby(data, :stepsize) begin
            @combine(:lpd_ci   = run_bootstrap(:lpd))
        end
        h  = data′[:,:stepsize]
            
        lpd      = data′[:,:lpd_ci]
        lpd_mean = [lpdᵢ[1] for lpdᵢ ∈ lpd]
        lpd_p    = [abs(lpdᵢ[2] - lpdᵢ[1]) for lpdᵢ ∈ lpd]
        lpd_m    = [abs(lpdᵢ[3] - lpdᵢ[1]) for lpdᵢ ∈ lpd]

                println(lpd_p, " ", lpd_m)

        write(h5, "h_$(dataset)",    h)
        write(h5, "rmse_$(dataset)", hcat(lpd_mean, lpd_p, lpd_m)' |> Array)
    end
end
