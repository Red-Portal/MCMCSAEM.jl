
include("common.jl")

struct PharmaNLME{
    V   <: AbstractVector,
    I   <: AbstractVector{<:Integer},
    Ind <: AbstractRange,
}
    t        ::V
    dosage   ::V
    weight   ::V

    y        ::V
    subject  ::I

    n_subject::Int

    ka_idx ::Int
    vol_idx::Ind
    clr_idx::Ind
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
    vol_idx   = 1:n_subject
    clr_idx   = n_subject + 1:2*n_subject
    ka_idx    = 2*n_subject + 1

    PharmaNLME(
        t, dosage, weight, y, subject, n_subject,
        ka_idx, vol_idx, clr_idx
    )
end

function LogDensityProblems.dimension(model::PharmaNLME)
    SimpleUnPack.@unpack n_subject, y = model
    2*n_subject + 1
end

function LogDensityProblems.capabilities(::Type{<:PharmaNLME})
    LogDensityProblems.LogDensityOrder{0}()
end

function model1cptmt(ka::Real, k::Real, dos::Real, vol::Real, t::Real)
    dos*ka/(vol*(ka - k))*(exp(-k*t) - exp(-ka*t))
end

function LogDensityProblems.logdensity(
    model::PharmaNLME, η::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack weight, dosage, t, y, subject, n_subject, vol_idx, clr_idx, ka_idx = model

    η_vol = η[vol_idx]
    η_clr = η[clr_idx]
    η_ka  = η[ka_idx]

    ℓμ_ka  = θ[1]
    ℓμ_vol = θ[2]
    ℓμ_clr = θ[3]

    σ_ka  = θ[4]
    σ_vol = θ[5]
    σ_clr = θ[6]

    β     = θ[7]
    a     = θ[8]

    vol = @. exp(ℓμ_vol + η_vol)
    ka  = @. exp(ℓμ_ka  + η_ka)
    clr = @. exp(ℓμ_clr + β*weight + η_clr)

    ℓp_η_ℓvol = logpdf(MvNormal(Zeros(n_subject), σ_vol), η_vol)
    ℓp_η_ℓclr = logpdf(MvNormal(Zeros(n_subject), σ_clr), η_clr)
    ℓp_η_ℓka  = logpdf(Normal(  ℓμ_ka,            σ_ka ),  η_ka)
    
    k       = clr./vol
    k_vec   = k[     subject]
    dos_vec = dosage[subject]
    vol_vec = vol[   subject]

    μ    = @. model1cptmt(ka, k_vec, dos_vec, vol_vec, t)

    #println(sqrt(mean(abs2, μ[1:10] - y[1:10])))

    ℓp_y = logpdf(MvNormal(μ, a), y)

    ℓp_y + ℓp_η_ℓvol + ℓp_η_ℓclr + ℓp_η_ℓka
end

function MCMCSAEM.sufficient_statistic(
    model::PharmaNLME,
    x    ::AbstractMatrix,
    θ    ::AbstractVector
)
    SimpleUnPack.@unpack t, y, dosage, subject, weight, vol_idx, clr_idx, ka_idx = model
    mean(eachcol(x)) do xi
        η_ka  = xi[ka_idx]
        η_vol = xi[vol_idx]
        η_clr = xi[clr_idx]

        β      = θ[7]
        ℓμ_ka  = θ[1]
        ℓμ_vol = θ[2]
        ℓμ_clr = θ[3]

        vol = @. exp(ℓμ_vol + η_vol)
        ka  = @. exp(ℓμ_ka  + η_ka)
        clr = @. exp(ℓμ_clr + β*weight + η_clr)

        k       = clr./vol
        k_vec   = k[     subject]
        dos_vec = dosage[subject]
        vol_vec = vol[   subject]
        μ       = @. model1cptmt(ka, k_vec, dos_vec, vol_vec, t)
        ϵ       = y - μ

        Ex_ka  = mean(η_ka)
        Ex_vol = mean(η_vol)
        Ex_clr = mean(η_clr)

        Ex2_ka  = mean(η_ka.^2)
        Ex2_vol = mean(η_vol.^2)
        Ex2_clr = mean(η_clr.^2)
        Eclrwht = mean(η_clr.*weight)

        Eϵ2 = mean(ϵ.^2)

        [Ex_ka, Ex2_ka, Ex_vol, Ex2_vol, Ex_clr, Ex2_clr, Eclrwht, Eϵ2]
    end
end

function MCMCSAEM.preconditioner(model::PharmaNLME, θ::AbstractVector)
    n_subject = model.n_subject
    ϵ  = 1e-7
    σ_ka, σ_vol, σ_ℓclr = θ[4], θ[5], θ[6]
    Diagonal(vcat(
        fill(σ_vol + ϵ, n_subject), fill(σ_ℓclr + ϵ, n_subject), [σ_ka + ϵ]
    ))
end

function MCMCSAEM.maximize_surrogate(model::PharmaNLME, S::AbstractVector)
    SimpleUnPack.@unpack weight, dosage, t, n_subject = model

    Ex_ka, Ex2_ka   = S[1], S[2]
    Ex_vol, Ex2_vol = S[3], S[4]
    Ex_clr, Ex2_clr = S[5], S[6]
    Eclrwht         = S[7]
    Eϵ2             = S[8]

    σ_ℓka  = sqrt(Ex2_ka  - Ex_ka^2)
    σ_ℓvol = sqrt(Ex2_vol - Ex_vol^2)
    a      = sqrt(Eϵ2)

    var_weight = var(weight)
    β          = 0 #(Eclrwht - Ex_clr*mean(weight))/var_weight
    μ_ℓclr     = Ex_clr #- β*mean(weight)
    σ_ℓclr     = sqrt(Ex2_clr - Ex_clr.^2) #0.5

    [ Ex_ka, Ex_vol, μ_ℓclr, σ_ℓka, σ_ℓvol, σ_ℓclr, β, a]
end

function viz(idx, ℓV, ℓclr, ℓka)
    #data = RDatasets.dataset("datasets", "Theoph")
    data = readdlm(datadir("theophylline_saemix.csv"), ',', Any, '\n')
    data = DataFrame(identity.(data), ["Subject", "Dose", "Time", "Conc", "Wt", "Sex"])
    intvl = 10

    n_subjects = length(unique(data.Subject))
    idx_range  = (idx-1)*intvl+1:idx*intvl
    t          = data.Time[idx_range] |> Vector
    Plots.scatter(t, data.Conc[idx_range], color=:red, yscale=:log10)

    map(ℓV, ℓclr, ℓka) do ℓV_i, ℓclr_i, ℓka_i
        ka  = exp(ℓka_i)
        dos = fill(data.Dose[(idx-1)*n_subjects + 1], intvl)
        vol = fill(exp(ℓV_i), intvl)
        clr = exp(ℓclr_i)
        k   = clr ./ vol
        μ   = @. model1cptmt(ka, k, dos, vol, t)
        Plots.plot!(t, μ, color=:blue, alpha=0.5, yscale=:log10) |> display
    end
end


function main(mcmc_type, h, key = 1)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    #ad = ADTypes.AutoReverseDiff()
    ad = ADTypes.AutoForwardDiff()

    #data = RDatasets.dataset("datasets", "Theoph")
    #intvl = 11

    data  = readdlm(datadir("theophylline_saemix.csv"), ',', Any, '\n')
    data  = DataFrame(identity.(data), ["Subject", "Dose", "Time", "Conc", "Wt", "Sex"])
    intvl = 10

    display(data)

    subjects = @. parse(Int, string(data.Subject))
    T        = Array(data.Time)
    D        = Array(data.Dose[1:intvl:end])
    BW       = Array(data.Wt[  1:intvl:end])
    y        = Array(data.Conc)

    model = PharmaNLME(D, BW, T, y, subjects)

    T_burn    = 1000
    T         = 10000
    γ₀        = 1e-1
    γ         = t -> γ₀/sqrt(t)

    θ₀ = [-1,0,0,0.1,1,1,0,1]
    x₀ = reshape(randn(rng, LogDensityProblems.dimension(model)), (:,1))

    function callback!(t, x, θ, stat)
        (
            μ_ka  = θ[1],
            μ_vol = θ[2],
            μ_clr = θ[3],
            σ_ka  = θ[4],
            σ_vol = θ[5],
            σ_clr = θ[6],
            a     = θ[8]
         )
    end

    θ, stats = MCMCSAEM.mcmcsaem(
        rng, model, x₀, θ₀, T, T_burn, γ, h;
        ad, callback! = callback!,
        show_progress = true,
        mcmc_type,
        n_inner_mcmc  = 4
    )
    stats_loglike = filter(Base.Fix2(haskey, :loglike), stats)
    Plots.plot([stat.loglike for stat in stats_loglike[2:end]]) |> display

    #θ = θ₀ #[log(1.58), log(31.6), log(1.55), 0.3, 0.02, 0.06, 0.008, 0.74]

    β_post = MCMCSAEM.mcmc(rng, model, θ, x₀, 1e-3, 4000; ad, show_progress = true)

    #@info("",
    #      mean(β_post[1,:]),
    #      mean(β_post[13,:]),
    #      mean(β_post[25,:]))

    #idx = 3
    #viz(idx, β_post[idx,1:100:end] .+ θ[2], β_post[12+idx,1:100:end] .+ θ[3], β_post[25,1:100:end] .+ θ[1])
end

