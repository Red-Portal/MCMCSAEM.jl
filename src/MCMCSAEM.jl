
module MCMCSAEM

using ADTypes
using DiffResults
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems
using ProgressMeter
using Random
using Statistics
using StatsFuns

using ForwardDiff, ReverseDiff, Zygote

function value_and_gradient! end

function sufficient_statistic end

sufficient_statistic(model, x, θ) = sufficient_statistic(model, x)

function maximize_surrogate end

preconditioner(model, θ) = I

include("gradient.jl")
include("mcmcsaem.jl")
include("mcmc.jl")

end
