# common.jl — shared constants, theme, data loading, and DMD pipeline
# include() this at the top of every Figure X.jl
# Guard: only execute the body once per Julia session.
if !@isdefined(COMMON_JL_LOADED)
const COMMON_JL_LOADED = true
ENV["QUARTO_FIG_FORMAT"] = "svg"
using Pkg; Pkg.activate(@__DIR__; io=devnull)
using LinearAlgebra, Statistics, Random, Serialization, Logging
using NPZ, CSV, DataFrames, Printf
using Plots, Colors
gr(dpi=150)

# ── Cache helpers ─────────────────────────────────────────────────────────────
const CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

cache_path(name) = joinpath(CACHE_DIR, name)

# Load from cache if file exists, otherwise evaluate expr, save, and return.
macro cached(file, expr)
    quote
        local _f = $(esc(file))
        if isfile(_f)
            deserialize(_f)
        else
            local _result = $(esc(expr))
            serialize(_f, _result)
            _result
        end
    end
end

# Thread-safe print lock
const PRINT_LOCK = ReentrantLock()

# ── Publication-ready plot defaults ───────────────────────────────────────────
# Figures are rendered at size= pixels then displayed at out-width:100%.
# Font sizes tuned for ~300px-wide panels at 150 dpi scaled to full page width.
const PUB_MARGIN      = 8Plots.mm
const PUB_GUIDE_FS    = 9
const PUB_TICK_FS     = 8
const PUB_LEGEND_FS   = 8
const PUB_TITLE_FS    = 10
const PUB_LW          = 2.0
const PUB_MSIZ        = 4

const PUB_W  = 300   # px per panel (1- and 2-panel figures)
const PUB_H  = 220   # px per panel height
const PUB_W3 = 420   # px per panel for 3-panel figures (more room for labels)

pub_theme!() = theme(:default;
    margin        = PUB_MARGIN,
    guidefontsize  = PUB_GUIDE_FS,
    tickfontsize   = PUB_TICK_FS,
    legendfontsize = PUB_LEGEND_FS,
    titlefontsize  = PUB_TITLE_FS,
    dpi            = 150,
)
pub_theme!()

function pub_plot(args...; kw...)
    plot(args...;
         margin        = PUB_MARGIN,
         guidefontsize  = PUB_GUIDE_FS,
         tickfontsize   = PUB_TICK_FS,
         legendfontsize = PUB_LEGEND_FS,
         titlefontsize  = PUB_TITLE_FS,
         size           = (PUB_W, PUB_H),
         kw...)
end

# ── Global colour / group settings ────────────────────────────────────────────
const GROUP_ORDER  = ["AB", "HF", "LF"]
const GROUP_COLORS = Dict(
    "AB" => RGB(0.27, 0.51, 0.71),   # steelblue
    "HF" => RGB(1.00, 0.55, 0.00),   # darkorange
    "LF" => RGB(0.86, 0.08, 0.24),   # crimson
)

# ── Data paths ─────────────────────────────────────────────────────────────────
const LEGACY_DATA_DIR = "/home/michael/Synology/Julia/data"
const REPO_DATA_DIR   = joinpath(@__DIR__, "data")
const DATA_DIR = get(ENV, "GAIT_DATA_DIR",
                     isdir(REPO_DATA_DIR) ? REPO_DATA_DIR : LEGACY_DATA_DIR)
const DATA_PATH = joinpath(DATA_DIR, "all_human_data.npy")
const META_PATH = joinpath(DATA_DIR, "all_human_data_metadata.csv")

function load_gait_data()
    if !isfile(DATA_PATH) || !isfile(META_PATH)
        missing = String[]
        !isfile(DATA_PATH) && push!(missing, DATA_PATH)
        !isfile(META_PATH) && push!(missing, META_PATH)
        msg = "Missing required gait data file(s):\n  " * join(missing, "\n  ") *
              "\nSet GAIT_DATA_DIR to a directory containing all_human_data.npy and all_human_data_metadata.csv"
        error(msg)
    end
    data      = npzread(DATA_PATH)
    meta      = CSV.read(META_PATH, DataFrame)
    return data, meta[!, "speed"], meta[!, "lf_or_hf"], meta[!, "subject"]
end

# ── Hankel DMD pipeline ────────────────────────────────────────────────────────
const FS              = 100.0
const TAU_GAIT        = 10
const NOISE_FLOOR     = 1e-6
const RESOLVENT_FLOOR = 1e-12

# Build Hankel matrix from trial_data (n_t × n_joints), τ lags.
# Returns (n_joints*(τ+1)) × (n_t - τ).
function build_hankel_multi(trial_data::AbstractMatrix, τ::Int)
    n_t, n_j = size(trial_data)
    H = zeros(n_j * (τ + 1), n_t - τ)
    for col in 1:(n_t - τ), lag in 0:τ
        H[lag*n_j+1 : (lag+1)*n_j, col] = trial_data[col + lag, :]
    end
    return H
end

# Stability-enforced DMD operator; eigenvalues clamped to unit disk.
# Returns (A_reduced, U_r, rank).
function get_stable_dmd_operator(trial_data::AbstractMatrix;
                                  τ::Int      = TAU_GAIT,
                                  nf::Float64 = NOISE_FLOOR)
    H        = build_hankel_multi(trial_data, τ)
    X, Y     = H[:, 1:end-1], H[:, 2:end]
    U, Σ, Vt = svd(X)
    r        = max(2, count(Σ .> Σ[1] * nf))
    Ã        = U[:, 1:r]' * Y * Vt[:, 1:r] * Diagonal(1.0 ./ Σ[1:r])
    Λ, W     = eigen(Ã)
    for i in eachindex(Λ)
        abs(Λ[i]) > 1.0 && (Λ[i] /= abs(Λ[i]))
    end
    return real.(W * Diagonal(Λ) * inv(W)), U[:, 1:r], r
end

# Henrici non-normality: ||A'A - AA'||_2 / ||A||_2^2
function henrici_departure(A::AbstractMatrix)
    opnorm(A' * A - A * A', 2) / opnorm(A, 2)^2
end

# Resolvent norm swept around the unit circle; returns (freq_hz, gains).
function resolvent_gain_on_circle(A::AbstractMatrix; n_angles=2000, fs=FS)
    thetas  = range(0, 2π, length=n_angles+1)[1:end-1]
    n       = size(A, 1)
    gains   = zeros(n_angles)
    Threads.@threads for i in 1:n_angles
        z        = exp(im * thetas[i])
        gains[i] = 1.0 / max(minimum(svdvals(z * I(n) - A)), RESOLVENT_FLOOR)
    end
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq     = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

# Dominant peak frequency above dc_cutoff_hz.
function find_stride_freq(freq_hz, gains; dc_cutoff_hz=0.3)
    mask = freq_hz .>= dc_cutoff_hz
    any(mask) || return freq_hz[argmax(gains)]
    idx = findall(mask)
    return freq_hz[idx[argmax(gains[idx])]]
end

# Speed-matching helper.
function speed_match_flag(speed_all, group_all)
    max_hf    = maximum(speed_all[group_all .== "HF"])
    ab_mask   = group_all .== "AB"
    keep_mask = .!ab_mask .| (speed_all .<= max_hf)
    return keep_mask, max_hf
end

# ── Hierarchical bootstrap helpers ────────────────────────────────────────────
# Resample subjects first, then trials within the sampled subject. This respects
# the repeated-trial structure of the gait dataset while allowing unequal trial
# counts per subject.
function hierarchical_resample_indices(subjects; rng=Random.default_rng())
    subj_levels = unique(subjects)
    sampled_subj = rand(rng, subj_levels, length(subj_levels))
    boot_idx = Int[]
    for subj in sampled_subj
        idx = findall(isequal(subj), subjects)
        append!(boot_idx, rand(rng, idx, length(idx)))
    end
    return boot_idx
end

bootstrap_ci(boot_stats; level=0.95) =
    quantile(boot_stats, [(1 - level) / 2, 1 - (1 - level) / 2])

bootstrap_pvalue(boot_stats) =
    2 * min(mean(boot_stats .<= 0), mean(boot_stats .>= 0))

function hierarchical_bootstrap_one(values, subjects;
                                    statfun=median,
                                    n_boot=4000,
                                    seed=42)
    rng = MersenneTwister(seed)
    point = statfun(values)
    boot_stats = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        idx = hierarchical_resample_indices(subjects; rng=rng)
        boot_stats[b] = statfun(values[idx])
    end
    return (point=point, ci=bootstrap_ci(boot_stats), p=bootstrap_pvalue(boot_stats),
            boot=boot_stats)
end

function hierarchical_bootstrap_diff(values_a, subjects_a, values_b, subjects_b;
                                     statfun=median,
                                     n_boot=4000,
                                     seed=42)
    rng = MersenneTwister(seed)
    point = statfun(values_a) - statfun(values_b)
    boot_stats = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        ia = hierarchical_resample_indices(subjects_a; rng=rng)
        ib = hierarchical_resample_indices(subjects_b; rng=rng)
        boot_stats[b] = statfun(values_a[ia]) - statfun(values_b[ib])
    end
    return (point=point, ci=bootstrap_ci(boot_stats), p=bootstrap_pvalue(boot_stats),
            boot=boot_stats)
end

function hierarchical_bootstrap_corr(x, y, subjects;
                                     n_boot=4000,
                                     seed=42)
    rng = MersenneTwister(seed)
    point = cor(x, y)
    boot_stats = Float64[]
    sizehint!(boot_stats, n_boot)
    for _ in 1:n_boot
        idx = hierarchical_resample_indices(subjects; rng=rng)
        xb, yb = x[idx], y[idx]
        (std(xb) == 0 || std(yb) == 0) && continue
        push!(boot_stats, cor(xb, yb))
    end
    return (point=point, ci=bootstrap_ci(boot_stats), p=bootstrap_pvalue(boot_stats),
            boot=boot_stats)
end

function hierarchical_bootstrap_corr_diff(xa, ya, subjects_a, xb, yb, subjects_b;
                                          n_boot=4000,
                                          seed=42)
    rng = MersenneTwister(seed)
    point = cor(xa, ya) - cor(xb, yb)
    boot_stats = Float64[]
    sizehint!(boot_stats, n_boot)
    for _ in 1:n_boot
        ia = hierarchical_resample_indices(subjects_a; rng=rng)
        ib = hierarchical_resample_indices(subjects_b; rng=rng)
        xa_b, ya_b = xa[ia], ya[ia]
        xb_b, yb_b = xb[ib], yb[ib]
        any(std.((xa_b, ya_b, xb_b, yb_b)) .== 0) && continue
        push!(boot_stats, cor(xa_b, ya_b) - cor(xb_b, yb_b))
    end
    return (point=point, ci=bootstrap_ci(boot_stats), p=bootstrap_pvalue(boot_stats),
            boot=boot_stats)
end

fmt_pvalue(p) = p < 1e-3 ? "< 0.001" : "= $(round(p; digits=3))"

end # if !@isdefined(COMMON_JL_LOADED)
