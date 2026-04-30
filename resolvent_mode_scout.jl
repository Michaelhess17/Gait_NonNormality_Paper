#!/usr/bin/env julia
# resolvent_mode_scout.jl — Quick-look resolvent forcing/response modes across speeds
#
# Usage examples:
#   julia resolvent_mode_scout.jl
#   julia resolvent_mode_scout.jl --subject S01 --n 6
#   julia resolvent_mode_scout.jl --subject S01 --harmonic 2
#
# Outputs:
#   figures/resolvent_mode_scout_<subject>_response.svg
#   figures/resolvent_mode_scout_<subject>_forcing.svg

include("common.jl")
using LinearAlgebra, Statistics, Printf
using DataFrames, CSV
using Plots

# ── Small CLI helper ─────────────────────────────────────────────────────────
function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = replace(a[3:end], "-" => "_")
            if i < length(args) && !startswith(args[i+1], "--")
                d[key] = args[i+1]
                i += 2
            else
                d[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return d
end

function getint(d, k, default)
    haskey(d, k) || return default
    try
        return parse(Int, d[k])
    catch
        return default
    end
end

function getfloat(d, k, default)
    haskey(d, k) || return default
    try
        return parse(Float64, d[k])
    catch
        return default
    end
end

function getstr(d, k, default)
    haskey(d, k) || return default
    return String(d[k])
end

# Single-threaded sweep (mirrors Figure 3 logic) for stride frequency picking.
function resolvent_gain_serial(A::AbstractMatrix; n_angles::Int=1000, fs::Float64=FS)
    thetas = range(0, 2π, length=n_angles+1)[1:end-1]
    n      = size(A, 1)
    gains  = Vector{Float64}(undef, length(thetas))
    for (i, θ) in enumerate(thetas)
        z = exp(im * θ)
        gains[i] = 1.0 / max(minimum(svdvals(z * I(n) - A)), RESOLVENT_FLOOR)
    end
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq     = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

function peak_freq_in_band(freq_hz::AbstractVector, gains::AbstractVector;
                           fmin::Float64=0.3, fmax::Float64=Inf)
    mask = (freq_hz .>= fmin) .& (freq_hz .<= fmax) .& .!isnan.(gains)
    any(mask) || return freq_hz[argmax(gains)]
    idx = findall(mask)
    return freq_hz[idx[argmax(gains[idx])]]
end

# Compute leading resolvent forcing/response modes at frequency z.
# We compute the SVD of M = (zI - A), then modes for H=M^{-1} follow as:
#   response = V[:, end], forcing = U[:, end], gain = 1/σ_min(M)
function leading_resolvent_modes(A::AbstractMatrix, z::ComplexF64)
    n = size(A, 1)
    M = z * I(n) - A
    F = svd(M)
    σmin = max(F.S[end], RESOLVENT_FLOOR)
    gain = 1.0 / σmin
    forcing  = F.U[:, end]
    response = F.V[:, end]
    return gain, forcing, response
end

# Input-output resolvent where inputs act on the lag-0 joint block and outputs
# read out the lag-0 joint block. Modes are returned in joint coordinates.
function leading_io_resolvent_modes(A::AbstractMatrix, U::AbstractMatrix, z::ComplexF64,
                                   n_joints::Int)
    n_full = size(U, 1)
    r      = size(U, 2)
    n_full >= n_joints || error("U rows < n_joints")

    # B_full injects into lag-0 joint coordinates of the full Hankel state.
    B_full = zeros(ComplexF64, n_full, n_joints)
    @inbounds for j in 1:n_joints
        B_full[j, j] = 1.0 + 0im
    end
    B_red = ComplexF64.(U') * B_full                 # r × n_joints
    C_red = ComplexF64.(U[1:n_joints, :])            # n_joints × r

    # Transfer function: G = C (zI - A)^{-1} B
    H = (z * I(r) - ComplexF64.(A)) \ B_red          # r × n_joints
    G = C_red * H                                   # n_joints × n_joints

    S = svd(G)
    gain     = S.S[1]
    forcing  = S.V[:, 1]
    response = S.U[:, 1]
    return gain, forcing, response
end

# Convert a Hankel-space mode vector into (lag × joint) magnitude matrix.
function mode_mag_lag_joint(mode_full::AbstractVector, n_joints::Int, τ::Int)
    length(mode_full) == n_joints * (τ + 1) || error("Unexpected mode size")
    M = reshape(mode_full, n_joints, τ + 1)'  # (τ+1) × n_joints
    return abs.(M)
end

function unit_norm(v)
    nrm = norm(v)
    nrm == 0 && return v
    return v ./ nrm
end

function phase_align_to(ref::AbstractVector{<:Complex}, v::AbstractVector{<:Complex})
    # Multiply v by a unit-modulus complex scalar so that <ref, v> is real-positive.
    a = dot(conj(ref), v)
    abs(a) == 0 && return v
    return v * exp(-im * angle(a))
end

function maxabs_norm(v)
    m = maximum(abs.(v))
    m == 0 && return v
    return v ./ m
end

function pick_spread_indices(speeds::AbstractVector{<:Real}, n_keep::Int)
    n = length(speeds)
    n_keep = min(n_keep, n)
    n_keep <= 1 && return [1]
    qs = range(0.0, 1.0, length=n_keep)
    return unique([clamp(round(Int, 1 + q * (n - 1)), 1, n) for q in qs])
end

# ── Main ─────────────────────────────────────────────────────────────────────
args = parse_args(ARGS)
subject_arg = get(args, "subject", "")
N_SHOW      = getint(args, "n", 6)
HARMONIC    = getint(args, "harmonic", 1)
N_ANGLES    = getint(args, "angles", 1000)
fmin_hz     = getfloat(args, "fmin_hz", 0.3)
fmax_hz     = getfloat(args, "fmax_hz", 5.0)
dc_cutoff   = getfloat(args, "dc_cutoff_hz", 0.3)
freq_mode   = lowercase(getstr(args, "freq", "stride")) # stride | zero | peak | given
f_given_hz  = getfloat(args, "f_hz", NaN)
ref_speed   = getfloat(args, "ref_speed", NaN)          # nearest selected speed becomes reference
out_tag     = getstr(args, "out_tag", "")
NO_FIG      = haskey(args, "no_fig")
IO_MODE     = haskey(args, "io")

println("Loading data…")
data, speed_all, group_all, subj_all = load_gait_data()

subjects = collect(subj_all)
subj_levels = unique(subjects)

# Default subject: most trials (tends to be a good first look)
if isempty(subject_arg)
    counts = Dict(s => sum(subjects .== s) for s in subj_levels)
    subject = argmax(collect(values(counts)))
    # argmax returns index in values; recover key
    subj_keys = collect(keys(counts))
    subj_vals = collect(values(counts))
    subject = subj_keys[argmax(subj_vals)]
else
    subject = subject_arg
end

idx_all = findall(isequal(subject), subjects)
if isempty(idx_all)
    error("No trials found for subject='$subject'. Try: julia resolvent_mode_scout.jl --subject <ID>\nAvailable subjects (first 10): $(join(subj_levels[1:min(end,10)], ", "))")
end

speeds_subj = Float64.(speed_all[idx_all])
groups_subj = String.(group_all[idx_all])

# Basic sanity: subject should not mix groups
g_unique = unique(groups_subj)
subj_group = length(g_unique) == 1 ? g_unique[1] : "mixed"

# Sort trials by speed
ord = sortperm(speeds_subj)
idx_sorted = idx_all[ord]
speed_sorted = speeds_subj[ord]

sel_pos = pick_spread_indices(speed_sorted, N_SHOW)
sel_idx = idx_sorted[sel_pos]
sel_speed = speed_sorted[sel_pos]

println("Subject: $subject   group: $subj_group   trials: $(length(idx_all))")
println("Selected $(length(sel_idx)) trials spanning speeds [$(minimum(speed_sorted)), $(maximum(speed_sorted))]")
println("Frequency selection: freq=$freq_mode  band=[$fmin_hz, $fmax_hz] Hz  dc_cutoff=$dc_cutoff Hz  harmonic=$HARMONIC")
println("Mode type: ", IO_MODE ? "input-output (lag0→lag0)" : "full-state")
if freq_mode == "given"
    isfinite(f_given_hz) || error("--freq given requires --f_hz <value>")
    @printf("Using fixed frequency f=%.3f Hz for all trials.\n", f_given_hz)
end

# Infer n_joints and τ from Hankel embedding settings
τ = TAU_GAIT
n_t, n_joints = size(data[sel_idx[1], :, :])

# Compute modes per selected trial
resp_fulls = Vector{Vector{ComplexF64}}()
forc_fulls = Vector{Vector{ComplexF64}}()
mode_gains = Float64[]
f_peak_v   = Float64[]
f_stride_v = Float64[]
f_used_v   = Float64[]

# First pass: compute per-trial peak and stride frequencies (for printing and selection).
for (k, i_trial) in enumerate(sel_idx)
    trial = data[i_trial, :, :]
    try
        A, _, _ = get_stable_dmd_operator(trial)
        fh, gh  = resolvent_gain_serial(A; n_angles=N_ANGLES)
        f_peak  = peak_freq_in_band(fh, gh; fmin=fmin_hz, fmax=fmax_hz)
        f_stride = find_stride_freq(fh, gh; dc_cutoff_hz=dc_cutoff)
        # If stride pick lands outside the band, fall back to band-limited peak
        if isfinite(fmax_hz) && f_stride > fmax_hz
            f_stride = f_peak
        end
        push!(f_peak_v, f_peak)
        push!(f_stride_v, f_stride)
    catch
        push!(f_peak_v, NaN)
        push!(f_stride_v, NaN)
    end
end

for (k, i_trial) in enumerate(sel_idx)
    trial = data[i_trial, :, :]
    try
        A, U, _ = get_stable_dmd_operator(trial)
        f_peak   = f_peak_v[k]
        f_stride = f_stride_v[k]
        f_used = if freq_mode == "stride"
            f_stride
        elseif freq_mode == "peak"
            f_peak
        elseif freq_mode == "zero"
            0.0
        elseif freq_mode == "given"
            f_given_hz
        else
            error("Unknown --freq mode '$freq_mode' (use stride|zero|peak|given)")
        end
        z = exp(im * 2π * (f_used * HARMONIC) / FS)
        if IO_MODE
            gain, forcing_j, response_j = leading_io_resolvent_modes(A, U, z, n_joints)
            push!(forc_fulls, ComplexF64.(forcing_j))
            push!(resp_fulls, ComplexF64.(response_j))
        else
            gain, forcing_r, response_r = leading_resolvent_modes(A, z)
            forcing_full  = ComplexF64.(U * forcing_r)
            response_full = ComplexF64.(U * response_r)
            push!(forc_fulls, forcing_full)
            push!(resp_fulls, response_full)
        end
        push!(mode_gains, gain)
        push!(f_used_v, f_used)

        @printf("  [%d/%d] speed=%6.1f  f_used=%.2f Hz (stride %.2f, peak %.2f)  gain(h=%d)=%.2e\n",
                k, length(sel_idx), sel_speed[k], f_used, f_stride, f_peak, HARMONIC, gain)
    catch err
        @printf("  [%d/%d] speed=%6.1f  FAILED (%s)\n", k, length(sel_idx), sel_speed[k], sprint(showerror, err))
        push!(forc_fulls, fill(0.0 + 0.0im, IO_MODE ? n_joints : n_joints*(τ+1)))
        push!(resp_fulls, fill(0.0 + 0.0im, IO_MODE ? n_joints : n_joints*(τ+1)))
        push!(mode_gains, NaN)
        push!(f_used_v, NaN)
    end
end

# Choose reference trial for similarity: default slowest selected speed.
ref_k = 1
if isfinite(ref_speed)
    ref_k = argmin(abs.(sel_speed .- ref_speed))
end
@printf("Reference for similarity: speed=%.1f (k=%d)\n", sel_speed[ref_k], ref_k)
println("Similarity metric: |<v_ref, v>| using unit-norm modes (phase-invariant cosine similarity)")

# Similarity vs speed (phase-invariant): abs(inner product)
ref = unit_norm(resp_fulls[ref_k])
sim_full = [abs(dot(conj(ref), unit_norm(v))) for v in resp_fulls]

if IO_MODE
    ref0 = unit_norm(resp_fulls[ref_k])
    sim_lag0 = [abs(dot(conj(ref0), unit_norm(v))) for v in resp_fulls]
else
    ref0 = unit_norm(resp_fulls[ref_k][1:n_joints])
    sim_lag0 = [abs(dot(conj(ref0), unit_norm(v[1:n_joints]))) for v in resp_fulls]
end

forc_ref_sim = unit_norm(forc_fulls[ref_k])
sim_forc_full = [abs(dot(conj(forc_ref_sim), unit_norm(v))) for v in forc_fulls]
if IO_MODE
    forc_ref0 = unit_norm(forc_fulls[ref_k])
    sim_forc_lag0 = [abs(dot(conj(forc_ref0), unit_norm(v))) for v in forc_fulls]
else
    forc_ref0 = unit_norm(forc_fulls[ref_k][1:n_joints])
    sim_forc_lag0 = [abs(dot(conj(forc_ref0), unit_norm(v[1:n_joints]))) for v in forc_fulls]
end

# Phase-aligned signed lag-0 vectors (more interpretable than magnitude heatmaps)
resp_ref = unit_norm(resp_fulls[ref_k])
forc_ref = unit_norm(forc_fulls[ref_k])

resp_lag0_signed = Vector{Vector{Float64}}(undef, length(resp_fulls))
forc_lag0_signed = Vector{Vector{Float64}}(undef, length(forc_fulls))
for k in 1:length(resp_fulls)
    r_al = phase_align_to(resp_ref, unit_norm(resp_fulls[k]))
    f_al = phase_align_to(forc_ref, unit_norm(forc_fulls[k]))
    if IO_MODE
        resp_lag0_signed[k] = collect(maxabs_norm(real.(r_al)))
        forc_lag0_signed[k] = collect(maxabs_norm(real.(f_al)))
    else
        resp_lag0_signed[k] = collect(maxabs_norm(real.(r_al[1:n_joints])))
        forc_lag0_signed[k] = collect(maxabs_norm(real.(f_al[1:n_joints])))
    end
end

# Pairwise similarity matrix (helps see clusters without picking one reference)
S = fill(NaN, length(resp_fulls), length(resp_fulls))
for i in 1:length(resp_fulls), j in 1:length(resp_fulls)
    vi = unit_norm(resp_fulls[i])
    vj = unit_norm(resp_fulls[j])
    S[i, j] = abs(dot(conj(vi), vj))
end

# ── Export a compact numeric summary for interpretation ─────────────────────
mkpath("figures")
tag = isempty(out_tag) ? @sprintf("%s_h%d", freq_mode, HARMONIC) : out_tag
tag = IO_MODE ? (tag * "_io") : tag
out_csv  = @sprintf("figures/resolvent_mode_scout_%s_%s_summary.csv", subject, tag)

df = DataFrame(
    k = collect(1:length(sel_speed)),
    trial_index = sel_idx,
    speed = sel_speed,
    f_peak_hz = f_peak_v,
    f_stride_hz = f_stride_v,
    f_used_hz = f_used_v,
    gain = mode_gains,
    sim_resp_full = sim_full,
    sim_resp_lag0 = sim_lag0,
    sim_forc_full = sim_forc_full,
    sim_forc_lag0 = sim_forc_lag0,
)
for j in 1:n_joints
    df[!, Symbol(@sprintf("resp_j%d", j))] = [resp_lag0_signed[k][j] for k in 1:length(sel_speed)]
    df[!, Symbol(@sprintf("forc_j%d", j))] = [forc_lag0_signed[k][j] for k in 1:length(sel_speed)]
end
CSV.write(out_csv, df)
println("Saved: $out_csv")

ok = .!isnan.(df.gain)
if any(ok)
    @printf("Summary (non-NaN runs): median sim_resp_full=%.2f, median sim_forc_full=%.2f\n",
            median(df.sim_resp_full[ok]), median(df.sim_forc_full[ok]))
end

# ── Plot helper ──────────────────────────────────────────────────────────────
function heatmap_mode_panel(mode_full, speed; title_prefix="")
    mag = mode_mag_lag_joint(unit_norm(mode_full), n_joints, τ)
    heatmap(1:n_joints, 0:τ, mag;
        xlabel="Joint index",
        ylabel="Lag",
        title = @sprintf("%s speed=%.1f", title_prefix, speed),
        colorbar=false,
        framestyle=:box,
    )
end

function lag0_signed_panel(v::AbstractVector{<:Real}, speed; title_prefix="")
    pub_plot(1:length(v), v;
        xlabel="Joint index",
        ylabel="Aligned mode",
        ylims=(-1.05, 1.05),
        xticks=(1:length(v), string.(1:length(v))),
        legend=false,
        lw=PUB_LW,
        color=:black,
        title=@sprintf("%s speed=%.1f", title_prefix, speed),
        framestyle=:box,
    )
end

NO_FIG && exit()

# ── Build figures ────────────────────────────────────────────────────────────

p_sim = pub_plot(sel_speed, sim_full;
    label="full Hankel mode",
    lw=PUB_LW,
    marker=:circle,
    color=:black,
    xlabel="Speed",
    ylabel="Mode similarity",
    ylims=(0,1.05),
    title=@sprintf("Resolvent response mode stability across speed (subject %s)", subject),
)
plot!(p_sim, sel_speed, sim_lag0;
    label="lag-0 block",
    lw=PUB_LW,
    marker=:diamond,
    color=:gray35,
)

 p_S = heatmap(1:length(sel_speed), 1:length(sel_speed), S;
    xlabel="trial index (sorted by speed)",
    ylabel="trial index (sorted by speed)",
    title="Pairwise response-mode similarity",
    clim=(0,1),
    c=:viridis,
    colorbar_title="|cos|",
    size=(PUB_W, PUB_H),
)

# Lag-0 signed panels
lp_resp = [lag0_signed_panel(resp_lag0_signed[k], sel_speed[k]; title_prefix="Resp lag0")
           for k in 1:length(sel_idx)]
lp_forc = [lag0_signed_panel(forc_lag0_signed[k], sel_speed[k]; title_prefix="Forc lag0")
           for k in 1:length(sel_idx)]

# Choose a compact grid
n_pan = length(sel_idx)
ncols = min(3, n_pan)
nrows = Int(ceil(n_pan / ncols))

p_resp = plot(lp_resp...; layout=(nrows, ncols), size=(ncols*PUB_W, nrows*PUB_H))
p_forc = plot(lp_forc...; layout=(nrows, ncols), size=(ncols*PUB_W, nrows*PUB_H))

fig_resp = plot(plot(p_sim, p_S; layout=(1,2), size=(2*PUB_W, PUB_H)),
                p_resp;
                layout=@layout([a; b]),
                size=(max(2,ncols)*PUB_W, (nrows+2)*PUB_H))
fig_forc = plot(plot(p_sim, p_S; layout=(1,2), size=(2*PUB_W, PUB_H)),
                p_forc;
                layout=@layout([a; b]),
                size=(max(2,ncols)*PUB_W, (nrows+2)*PUB_H))

out_resp = @sprintf("figures/resolvent_mode_scout_%s_response.svg", subject)
out_forc = @sprintf("figures/resolvent_mode_scout_%s_forcing.svg", subject)

tag = isempty(out_tag) ? @sprintf("%s_h%d", freq_mode, HARMONIC) : out_tag
out_resp = @sprintf("figures/resolvent_mode_scout_%s_%s_response.svg", subject, tag)
out_forc = @sprintf("figures/resolvent_mode_scout_%s_%s_forcing.svg", subject, tag)

savefig(fig_resp, out_resp)
savefig(fig_forc, out_forc)
println("Saved: $out_resp")
println("Saved: $out_forc")
