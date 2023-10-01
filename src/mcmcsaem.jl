
function langevin_kernel(
    rng::Random.AbstractRNG,
    g,
    x  ::AbstractVector{F},
    h  ::Real
) where {F <: Real}
    d = length(x)
    ξ = sqrt(h)*randn(rng, F, d)
    x + h/2*g(x) + ξ
end

function markovchain_transition!(
    rng    ::Random.AbstractRNG,
    ad     ::ADTypes.AbstractADType,
    model,
    xs     ::AbstractArray,
    θ      ::AbstractVector,
    h      ::Real,
    ∇ℓ_buf
)
    ℓ(x′)  = LogDensityProblems.logdensity(model, x′, θ)
    ∇ℓ(x′) = value_and_gradient!(ad, ℓ, x′, ∇ℓ_buf) |> DiffResults.gradient
    mapslices(xs, dims=1) do x′
        langevin_kernel(rng, ∇ℓ, x′, h)
    end
end

function compute_update!(
    ad    ::ADTypes.AbstractADType,
    model,
    xs    ::AbstractArray,
    θ     ::AbstractVector,
    H_buf
)
    q(θ′) = begin
        mean(eachcol(xs)) do xᵢ
            LogDensityProblems.logdensity(model, xᵢ, θ′)
        end
    end
    value_and_gradient!(ad, q, θ, H_buf)
    DiffResults.gradient(H_buf), DiffResults.value(H_buf)
end

function mcmcsaem(
    rng        ::Random.AbstractRNG,
    model,
    x₀s        ::AbstractMatrix,
    θ₀         ::AbstractVector,
    T          ::Int,
    γ_schedule,
    h          ::Real;
    ad         ::ADTypes.AbstractADType = ADTypes.AutoZygote,
    callback! = nothing
)
    xs   = copy(x₀s)
    θ    = copy(θ₀)
    prog = Progress(T)

    H_buf  = DiffResults.DiffResult(zero(eltype(θ)), similar(θ))
    ∇ℓ_buf = DiffResults.DiffResult(zero(eltype(xs)), similar(xs, size(xs, 1)))

    for t = 1:T
        xs   = markovchain_transition!(rng, ad, model, xs, θ, h, ∇ℓ_buf)
        H, V = compute_update!(ad, model, xs, θ, H_buf)

        γₜ = γ_schedule(t)
        θ  = project(model, θ + γₜ*H)

        if !isnothing(callback!)
            callback!(t, xs, θ)
        end

        ProgressMeter.next!(prog; showvalues = [(:t,t), (:loglike,V)])
    end
end
