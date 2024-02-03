
using DrWatson
@quickactivate "scripts"

using ADTypes
using Accessors
using Bootstrap
using DataFrames, DataFramesMeta
using Distributions
using FillArrays 
using LogDensityProblems
using LinearAlgebra
using MAT
using Plots, StatsPlots
using ProgressMeter
using Random
using Random, Random123
using SimpleUnPack
using Statistics
using StatsFuns
using Tullio
using HDF5
using JLD2

using MCMCSAEM

function prepare_dataset(rng   ::Random.AbstractRNG,
                         data_x::AbstractMatrix,
                         data_y::AbstractVector;
                         ratio ::Real = 0.9)
    n_data      = size(data_x, 1)
    shuffle_idx = Random.shuffle(rng, 1:n_data)
    data_x      = data_x[shuffle_idx,:]
    data_y      = data_y[shuffle_idx]

    n_train = floor(Int, n_data*ratio)
    x_train = data_x[1:n_train, :]
    y_train = data_y[1:n_train]
    x_test  = data_x[n_train+1:end, :]
    y_test  = data_y[n_train+1:end]
    x_train, y_train, x_test, y_test
end

function run_bootstrap(data′)
    boot = bootstrap(mean, data′, BalancedSampling(1024))
    confint(boot, PercentileConfInt(0.8)) |> only
end
