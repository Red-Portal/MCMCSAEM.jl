
struct BNN{
    F <: Real,
    V <: AbstractVector,
    M <: AbstractMatrix,
    R
}
    X::M
    y::V
    n_hidden::Int
    likeadj::F
    restruct::R
end

@functor BNN (X, y)

struct BNNParams{
    F <: Real,
    V <: AbstractVector{F},
    M <: AbstractMatrix{F}
}
    η_λ⁻¹::V
    η_γ⁻¹::V
    W₁   ::M
    W₂   ::M
end

@functor BNNParams

function BNN(X, y, n_hidden)
    X_pad      = vcat(X, ones(eltype(X), 1, size(X, 2)))
    n_features = size(X_pad,1)
    params     = BNNParams(
        similar(X_pad, 1),
        similar(X_pad, 1),
        similar(X_pad, n_hidden, n_features),
        similar(X_pad, 1, n_hidden+1)
    )
    _, re = Optimisers.destructure(params)
    BNN(X_pad, y, n_hidden, 1f0, re)
end

function LogDensityProblems.capabilities(::Type{<:BNN})
    LogDensityProblems.LogDensityOrder{0}()
end

function subsample_problem(prob::BNN, batch)
    @unpack X, y = prob
    prob′  = @set prob.X       = X[:,batch]
    prob′′  = @set prob′.y       = y[batch]
    prob′′′ = @set prob′′.likeadj = length(y)/length(batch)
    prob′′′
end

function logdensity(bnn::BNN, params::BNNParams{F,V,M}) where {F<:Real, V, M}
    @unpack X, y, likeadj = bnn
    @unpack W₁, W₂, η_λ⁻¹, η_γ⁻¹ = params

    b⁻¹ = Bijectors.bijector(Gamma(1,1)) |> inverse

    λ⁻¹, logabsJ_λ = with_logabsdet_jacobian(b⁻¹, only(η_λ⁻¹))
    γ⁻¹, logabsJ_γ = with_logabsdet_jacobian(b⁻¹, only(η_γ⁻¹))

    ℓp_λ⁻¹ = logpdf(InverseGamma{F}(F(6), F(1/6)), λ⁻¹)
    ℓp_γ⁻¹ = logpdf(InverseGamma{F}(F(6), F(1/6)), γ⁻¹)

    W₁_flat = reshape(W₁,:)
    W₂_flat = reshape(W₂,:)

    ℓp_W₁ = logpdf(MvNormal(Zeros(length(W₁_flat)), sqrt(λ⁻¹)), W₁_flat)
    ℓp_W₂ = logpdf(MvNormal(Zeros(length(W₂_flat)), sqrt(λ⁻¹)), W₂_flat)

    X2      = Flux.relu.(W₁*X / sqrt(size(X, 1)))
    X₂′      = vcat(X2, ones(1, size(X2, 2)))
    y′       = W₂*X₂′  / sqrt(size(X₂′, 1))
    y′_flat  = reshape(y′, :)

    ℓp_y = logpdf(MvNormal(y′_flat, sqrt.(γ⁻¹)), y)

    likeadj*ℓp_y + ℓp_W₁ + ℓp_W₂ + ℓp_λ⁻¹ + ℓp_γ⁻¹ +
        logabsJ_λ + logabsJ_γ
end

function LogDensityProblems.logdensity(model::BNN, θ::AbstractVector)
    logdensity(model, model.restruct(θ))
end

function LogDensityProblems.dimension(bnn::BNN)
    @unpack X, n_hidden  = bnn 
    n_features = size(X,1)
    d_W₁ = n_features*n_hidden
    d_W₂ = (n_hidden+1)*1
    d_W₁ + d_W₂ + 2
end

function propagate_linear(M, V, m_prev, v_prev)
    scaling = size(m_prev, 1)
    m_α     = M * m_prev / sqrt(scaling)
    v_α     = ((M.^2)*v_prev + V*(m_prev.^2) + V*v_prev) / scaling
    m_α, v_α
end

function propagate_relu(m_α, v_α)
    α  = m_α ./ sqrt.(v_α)
    γ  = normpdf.(-α) ./ normcdf.(α)

    unstable_idx    = α .< -30
    α_unstable      = α[unstable_idx]
    γ[unstable_idx] = -α_unstable - (1 ./ α_unstable) + (2 ./(α_unstable).^3)

    v′ = m_α + sqrt.(v_α).*γ

    m_b = StatsFuns.normcdf.(α) .* v′
    v_b = m_α .* v′ .* StatsFuns.normcdf.(-α) +
        StatsFuns.normcdf.(α) .* v_α .* (1 .- γ.*(γ .+ α))
    v_b = max.(v_b, 1e-12)

    m_b, v_b
end

function bnn_adf_predict(
    μ_W₁   ::AbstractMatrix,
    σ_W₁   ::AbstractMatrix,
    μ_W₂   ::AbstractMatrix,
    σ_W₂   ::AbstractMatrix,
    μ_η_γ⁻¹::Real,
    σ_η_γ⁻¹::Real,
    μ_y    ::Real,
    σ_y    ::Real,
    X
)
    # Heuristic moment matching approximation to the inverse-gamma distribution
    γ_α = 1 / (exp(σ_η_γ⁻¹^2) - 1) + 1
    γ_β = (γ_α - 1)*exp(μ_η_γ⁻¹ + σ_η_γ⁻¹^2/2)
   
    m_1, v_1 = propagate_linear(μ_W₁, σ_W₁.^2, X, zeros(size(X)))
    m_1, v_1 = propagate_relu(m_1, v_1)
    m_1      = vcat(m_1, ones( 1, size(m_1, 2)))
    v_1      = vcat(v_1, zeros(1, size(v_1, 2)))

    m_2, v_2 = propagate_linear(μ_W₂, σ_W₂.^2, m_1, v_1)
    m_y      = m_2.*σ_y .+ μ_y
    v_y      = v_2*σ_y.*σ_y
    m_y      = m_y[1,:]
    v_y      = v_y[1,:]
    v_noise  = γ_β/γ_α.*σ_y.*σ_y
    
    # ∫ Normal(y, σ²_y) ∫ Normal(0, 1/γ) × LogNormal(1/γ; μ, σ) dγ
    # ≈ ∫ Normal(y, σ²_y) ∫ Normal(0, 1/γ) × Gamma(γ; α, β)
    # = ∫ Normal(y, σ²_y) × TDist(ν=2α, μ=0, σ²=β/α)
    # ≈ ∫ Normal(y, σ²_y) × Normal(0, β/α)
    # = Normal(y, σ²_y+β/α)
    MvNormal(m_y, @. sqrt(v_y + v_noise))
end
