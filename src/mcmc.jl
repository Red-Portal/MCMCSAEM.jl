
function ula_kernel(
    rng::Random.AbstractRNG,
    g,
    x  ::AbstractVector{F},
    h  ::Real,
    P  = I
) where {F <: Real}
    d = length(x)
    ξ = rand(rng, MvNormal(Zeros{F}(d), h*P))
    x + h/2*P*g(x) + ξ
end

function mala_kernel(
    rng::Random.AbstractRNG,
    grad,
    x  ::AbstractVector{F},
    h  ::Real,
    P = I
) where {F <: Real}
    out = grad(x)
    ∇ℓπ = DiffResults.gradient(out)
    ℓπ  = DiffResults.value(out)

    μ  = x + h/2*P*∇ℓπ
    q  = MvNormal(μ, h*P)
    x′  = rand(rng, q)
    ℓq′ = logpdf(q, x′)

    out′ = grad(x′)
    ∇ℓπ′ = DiffResults.gradient(out′)
    ℓπ′  = DiffResults.value(out′)
    μ′   = x′ + h/2*P*∇ℓπ′
    q′   = MvNormal(μ′, h*P)
    ℓq  = logpdf(q′, x)

    ℓα = min(0, (ℓπ′ - ℓπ) - (ℓq′ - ℓq))
    if log(rand(rng)) < ℓα
        x′, exp(ℓα)
    else
        x, exp(ℓα)
    end
end

function mcmc_transition!(
    rng    ::Random.AbstractRNG,
    ad     ::ADTypes.AbstractADType,
           ::Val{:ula},
    model,
    x      ::AbstractArray,
    θ      ::AbstractVector,
    h      ::Real,
    ∇ℓ_buf
)
    ℓ(x′)  = LogDensityProblems.logdensity(model, x′, θ)
    ∇ℓ(x′) = value_and_gradient!(ad, ℓ, x′, ∇ℓ_buf) |> DiffResults.gradient
    P     = preconditioner(model, θ)
    xᵢ    = last(eachcol(x))
    for i = 1:size(x,2) 
        xᵢ     = ula_kernel(rng, ∇ℓ, xᵢ, h, P)
        x[:,i] = xᵢ
    end
    x, 1.0
end

function mcmc_transition!(
    rng    ::Random.AbstractRNG,
    ad     ::ADTypes.AbstractADType,
           ::Val{:mala},
    model,
    x      ::AbstractArray,
    θ      ::AbstractVector,
    h      ::Real,
    ∇ℓ_buf
)
    ℓ(x′)  = LogDensityProblems.logdensity(model, x′, θ)
    ∇ℓ(x′) = value_and_gradient!(ad, ℓ, x′, ∇ℓ_buf)
    P     = preconditioner(model, θ)
    acc   = 0
    xᵢ    = last(eachcol(x))
    for i = 1:size(x,2) 
        xᵢ, α′  = mala_kernel(rng, ∇ℓ, xᵢ, h, P)
        acc    += α′/size(x, 2)
        x[:,i]  = xᵢ
    end
    x, acc
end

function mcmc(rng    ::Random.AbstractRNG,
              model,
              θ      ::AbstractVector,
              x₀     ::AbstractArray,
              h₀     ::Real,
              T      ::Integer,
              T_adapt::Integer  = div(T,2);
              mcmc_type::Symbol = :mala,
              ad       ::ADTypes.AbstractADType = ADTypes.AutoZygote,
              show_progress = true)
    x      = x₀ isa Vector ? reshape(x₀, (:,1)) : copy(x₀)
    ∇ℓ_buf = DiffResults.DiffResult(zero(eltype(x)), similar(x, size(x, 1)))
    prog   = Progress(T; enabled=show_progress, showspeed=true)
    x_post = similar(x₀, size(x₀,1), T - T_adapt)

    if mcmc_type == :mala
        # Nesterov Dual Averaging
        acc    = 0
        H_dual = zeros(eltype(h₀), T_adapt)
        γ_dual = 0.05
        ℓh₀    = log(h₀)
        ℓh_bar = ℓh₀
        ℓh     = ℓh₀

        for t = 1:T_adapt
            x, α = mcmc_transition!(
                rng, ad, Val(mcmc_type), model, x, θ, exp(ℓh), ∇ℓ_buf
            )
            
            H_dual[t] = 0.47 - α
            ℓh        = ℓh₀ - sqrt(t)/γ_dual/(t + 10)*sum(H_dual[1:t])
            η         = 1/t^(0.5)
            ℓh_bar    = η*ℓh + (1 - η)*ℓh_bar
            
            acc = acc + (α - acc)/t
            
            stats = (
                t       = t,
                loglike = DiffResults.value(∇ℓ_buf),
                h       = exp(ℓh),
                ℓh_bar  = exp(ℓh_bar),
                acc     = acc,
                state   = :adaptation,
            )
            pm_next!(prog, stats)
        end
    end

    acc   = 0
    h_bar = exp(ℓh_bar)
    for t = 1:T-T_adapt
        x, α = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h_bar, ∇ℓ_buf)

        acc = acc + (α - acc)/t
        x_post[:,t] = x

        stats = (t=t, loglike=DiffResults.value(∇ℓ_buf), acc=acc, state = :sampling,)
        pm_next!(prog, stats)
    end
    x_post
end
