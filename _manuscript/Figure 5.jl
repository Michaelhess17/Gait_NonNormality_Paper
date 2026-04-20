#!/usr/bin/env julia
# Figure 5.jl — Exploratory mechanistic analyses for HF non-normality pattern
#
# Goal:
# 1) Explain why HF can have high Henrici departure (eta) despite lower rank and
#    faster normalized harmonic fall-off.
# 2) Quantify speed effects on dimensionality and related metrics (subject-specific slopes).
# 3) Link operator metrics to clinical function (Berg, Fugl-Meyer) at subject level.

include("common.jl")
using CSV, DataFrames, Printf, Serialization, Statistics, LinearAlgebra

const N_HARMONICS_F5 = 5
const N_ANGLES_F5 = 1000
const F5_BOOT = 2000

# ── Core metric helpers ───────────────────────────────────────────────────────
function resolvent_gain_serial(A::AbstractMatrix; n_angles=N_ANGLES_F5, fs=FS)
    thetas = range(0, 2π, length=n_angles+1)[1:end-1]
    n = size(A, 1)
    gains = [1.0 / max(minimum(svdvals(exp(im * θ) * I(n) - A)), RESOLVENT_FLOOR)
             for θ in thetas]
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

function resolvent_gain_full_circle(A::AbstractMatrix; n_angles=360)
    thetas = range(0, 2π, length=n_angles+1)[1:end-1]
    n = size(A, 1)
    gains = [1.0 / max(minimum(svdvals(exp(im * θ) * I(n) - A)), RESOLVENT_FLOOR)
             for θ in thetas]
    return collect(thetas), gains
end

function interp_linear(x::AbstractVector, y::AbstractVector, xq::Float64)
    (xq < x[1] || xq > x[end]) && return NaN
    j = searchsortedlast(x, xq)
    j >= length(x) && return y[end]
    x1, x2 = x[j], x[j+1]
    y1, y2 = y[j], y[j+1]
    x2 == x1 && return y1
    t = (xq - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)
end

function harmonic_gains_at(A::AbstractMatrix, f_stride::Float64;
                            n_harmonics=N_HARMONICS_F5, fs=FS)
    n = size(A, 1)
    map(1:n_harmonics) do h
        z = exp(im * 2π * (f_stride * h) / fs)
        1.0 / max(minimum(svdvals(z * I(n) - A)), RESOLVENT_FLOOR)
    end
end

function q_factor(A::AbstractMatrix, f_stride::Float64; fs=FS)
    n = size(A, 1)
    gf = f -> 1.0 / max(minimum(svdvals(exp(im * 2π * f / fs) * I(n) - A)), RESOLVENT_FLOOR)
    peak = gf(f_stride)
    thresh = peak / sqrt(2)
    step = 0.02

    fl = f_stride
    while fl > 0 && gf(fl) > thresh
        fl -= step
    end
    fr = f_stride
    while fr < fs / 2 && gf(fr) > thresh
        fr += step
    end

    bl_lo, bl_hi = fl, fl + step
    for _ in 1:6
        fm = (bl_lo + bl_hi) / 2
        gf(fm) > thresh ? (bl_hi = fm) : (bl_lo = fm)
    end
    br_lo, br_hi = fr - step, fr
    for _ in 1:6
        fm = (br_lo + br_hi) / 2
        gf(fm) > thresh ? (br_lo = fm) : (br_hi = fm)
    end

    bw = max((br_lo + br_hi) / 2 - (bl_lo + bl_hi) / 2, 1e-4)
    return f_stride / bw
end

function spectral_decay_rate(gnorms::AbstractVector)
    any(isnan, gnorms) && return NaN
    any(<=(0), gnorms) && return NaN
    y = log.(gnorms)
    x = collect(1.0:length(gnorms))
    xm, ym = mean(x), mean(y)
    denom = sum((x .- xm) .^ 2)
    denom == 0 && return NaN
    return sum((x .- xm) .* (y .- ym)) / denom
end

function harmonic_entropy(g::AbstractVector)
    any(isnan, g) && return NaN
    any(<=(0), g) && return NaN
    p = g ./ sum(g)
    return -sum(p .* log.(p))
end

function concentration_h1(g::AbstractVector)
    length(g) < 5 && return NaN
    any(isnan, g) && return NaN
    denom = mean(g[2:5])
    denom <= 0 && return NaN
    return g[1] / denom
end

# ── Stats helpers ─────────────────────────────────────────────────────────────
zscore_safe(x) = begin
    xf = collect(skipmissing(x))
    isempty(xf) && return fill(NaN, length(x))
    μ, σ = mean(xf), std(xf)
    σ == 0 && return fill(0.0, length(x))
    [(ismissing(v) || !isfinite(v)) ? NaN : (v - μ) / σ for v in x]
end

function ols_fit(X::AbstractMatrix, y::AbstractVector)
    β = X \ y
    yhat = X * β
    sst = sum((y .- mean(y)) .^ 2)
    r2 = sst == 0 ? NaN : 1 - sum((y .- yhat) .^ 2) / sst
    return β, r2
end

function try_parse_float(v)
    if v isa Number
        return Float64(v)
    end
    s = strip(String(v))
    isempty(s) && return NaN
    try
        return parse(Float64, s)
    catch
        return NaN
    end
end

function subject_resample_indices(subjects; rng=Random.default_rng())
    uniq = unique(subjects)
    sampled = rand(rng, uniq, length(uniq))
    idx = Int[]
    for s in sampled
        append!(idx, findall(==(s), subjects))
    end
    return idx
end

function intercept_slope(speed::AbstractVector, y::AbstractVector)
    ok = .!isnan.(speed) .& .!isnan.(y)
    x = speed[ok]
    v = y[ok]
    length(v) == 0 && return (NaN, NaN)
    length(v) == 1 && return (v[1], NaN)
    std(x) == 0 && return (mean(v), 0.0)
    X = hcat(ones(length(v)), x)
    β = X \ v
    return (β[1], β[2])
end

function partial_corr(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    ok = .!isnan.(x) .& .!isnan.(y) .& .!isnan.(z)
    xv, yv, zv = x[ok], y[ok], z[ok]
    length(xv) < 4 && return NaN
    std(xv) == 0 && return NaN
    std(yv) == 0 && return NaN
    std(zv) == 0 && return cor(xv, yv)
    Xz = hcat(ones(length(zv)), zv)
    rx = xv - Xz * (Xz \ xv)
    ry = yv - Xz * (Xz \ yv)
    std(rx) == 0 && return NaN
    std(ry) == 0 && return NaN
    return cor(rx, ry)
end

# ── Load data and compute per-trial metrics  [cached] ─────────────────────────
println("Loading data…")
data, speed_all, group_all, subj_all = load_gait_data()
n_trials = size(data, 1)

trial_metrics = @cached cache_path("fig5_trial_metrics.jls") begin
    println("Figure 5 trial metrics ($(Threads.nthreads()) threads)…")
    eta_v      = fill(NaN, n_trials)
    rank_v     = zeros(Int, n_trials)
    fstride_v  = fill(NaN, n_trials)
    eigdist_v  = fill(NaN, n_trials)
    qfac_v     = fill(NaN, n_trials)
    gains_v    = fill(NaN, n_trials, N_HARMONICS_F5)

    Threads.@threads for i in 1:n_trials
        try
            A, _, r = get_stable_dmd_operator(data[i, :, :])
            eta_v[i] = henrici_departure(A)
            rank_v[i] = r
            fh, gh = resolvent_gain_serial(A)
            fs_ = find_stride_freq(fh, gh)
            fstride_v[i] = fs_
            gains_v[i, :] = harmonic_gains_at(A, fs_)
            qfac_v[i] = q_factor(A, fs_)

            z = exp(im * 2π * fs_ / FS)
            eigdist_v[i] = minimum(abs.(eigvals(A) .- z))
        catch
        end
    end

    (eta=eta_v, rank=rank_v, fstride=fstride_v, eigdist=eigdist_v,
     qfac=qfac_v, gains=gains_v)
end

# ── Assemble trial DataFrame ──────────────────────────────────────────────────
df5 = DataFrame(subject = collect(subj_all),
                group = collect(group_all),
                speed = Float64.(speed_all),
                eta = trial_metrics.eta,
                rank = Float64.(trial_metrics.rank),
                f_stride = trial_metrics.fstride,
                eig_dist = trial_metrics.eigdist,
                qfactor = trial_metrics.qfac)

for h in 1:N_HARMONICS_F5
    df5[!, Symbol("gain_h$h")] = trial_metrics.gains[:, h]
end
for h in 1:N_HARMONICS_F5
    df5[!, Symbol("gnorm_h$h")] = trial_metrics.gains[:, h] ./ trial_metrics.gains[:, 1]
end

df5[!, :decay_rate] = [spectral_decay_rate(vec(trial_metrics.gains[i, :] ./ trial_metrics.gains[i, 1]))
                       for i in 1:n_trials]
df5[!, :c1] = [concentration_h1(vec(trial_metrics.gains[i, :])) for i in 1:n_trials]
df5[!, :harm_entropy] = [harmonic_entropy(vec(trial_metrics.gains[i, :])) for i in 1:n_trials]
df5[!, :log_eigdist] = log10.(max.(df5.eig_dist, 1e-12))
df5[!, :log_h1] = log10.(max.(df5.gain_h1, 1e-12))
df5[!, :log_h1_per_rank] = log10.(max.(df5.gain_h1 ./ max.(df5.rank, 1.0), 1e-12))
df5[!, :inv_eigdist] = 1.0 ./ max.(df5.eig_dist, 1e-12)
df5[!, :speed_matched] = speed_match_flag(df5.speed, df5.group)[1]

println("\n═══ Trial-level summary (speed-matched) ═══")
for g in GROUP_ORDER
    d = df5[(df5.group .== g) .& df5.speed_matched, :]
    η = median(filter(!isnan, d.eta))
    rk = median(filter(!isnan, d.rank))
    c1m = median(filter(!isnan, d.c1))
    dec = median(filter(!isnan, d.decay_rate))
    @printf("%-6s  eta=%.3f  rank=%.1f  C1=%.2f  decay=%.3f\n", g, η, rk, c1m, dec)
end

# ── Subject-level intercept/slope summaries ───────────────────────────────────
slope_metrics = [:eta, :rank, :decay_rate, :c1, :harm_entropy, :log_eigdist, :qfactor]
subjects = unique(df5.subject)
rows = Vector{NamedTuple}()
for s in subjects
    d = sort(df5[df5.subject .== s, :], :speed)
    g = d.group[1]
    row = Dict{Symbol, Any}(:subject => s, :group => g)
    for m in slope_metrics
        a, b = intercept_slope(d.speed, Float64.(d[!, m]))
        row[Symbol(string(m), "_int")] = a
        row[Symbol(string(m), "_slope")] = b
    end
    row[:n_trials] = nrow(d)
    push!(rows, (; row...))
end

df5_subj = DataFrame(rows)

println("\n═══ Subject-level slope summary (median by group) ═══")
for g in GROUP_ORDER
    d = df5_subj[df5_subj.group .== g, :]
    s_eta = median(filter(!isnan, d.eta_slope))
    s_rank = median(filter(!isnan, d.rank_slope))
    s_dec = median(filter(!isnan, d.decay_rate_slope))
    @printf("%-6s  eta_slope=%.4f  rank_slope=%.4f  decay_slope=%.4f\n", g, s_eta, s_rank, s_dec)
end

# ── Attenuation analysis: does HF effect shrink after adding mechanism vars? ──
# Outcome 1: normalized fall-off (decay_rate)
# Outcome 2: concentration (log C1)

function attenuation_models(df::DataFrame, ycol::Symbol; n_boot=F5_BOOT, seed=531)
    d = df[:, [:subject, :group, :speed, :eta, :rank, :log_eigdist, ycol]]
    rename!(d, ycol => :y)
    d = d[.!isnan.(d.y) .& .!isnan.(d.speed) .& .!isnan.(d.eta) .&
          .!isnan.(d.rank) .& .!isnan.(d.log_eigdist), :]
    if nrow(d) < 20
        return DataFrame()
    end

    d[!, :hf] = Float64.(d.group .== "HF")
    d[!, :lf] = Float64.(d.group .== "LF")
    d[!, :speed_z] = zscore_safe(d.speed)
    d[!, :eta_z] = zscore_safe(d.eta)
    d[!, :rank_z] = zscore_safe(d.rank)
    d[!, :eig_z] = zscore_safe(d.log_eigdist)

    function fit_hf_coef(dd, model_id)
        y = dd.y
        X = if model_id == 1
            hcat(ones(nrow(dd)), dd.hf, dd.lf, dd.speed_z)
        elseif model_id == 2
            hcat(ones(nrow(dd)), dd.hf, dd.lf, dd.speed_z, dd.rank_z)
        elseif model_id == 3
            hcat(ones(nrow(dd)), dd.hf, dd.lf, dd.speed_z, dd.rank_z, dd.eig_z)
        else
            hcat(ones(nrow(dd)), dd.hf, dd.lf, dd.speed_z, dd.rank_z, dd.eig_z, dd.eta_z)
        end
        β, r2 = ols_fit(X, y)
        return β[2], r2
    end

    out = DataFrame(model=String[], beta_hf=Float64[], ci_lo=Float64[], ci_hi=Float64[], r2=Float64[])
    labels = ["group+speed", "+rank", "+eigdist", "+eta"]
    rng = MersenneTwister(seed)

    for (mi, lab) in enumerate(labels)
        β0, r2 = fit_hf_coef(d, mi)
        boots = Float64[]
        sizehint!(boots, n_boot)
        for _ in 1:n_boot
            idx = subject_resample_indices(d.subject; rng=rng)
            db = d[idx, :]
            try
                βb, _ = fit_hf_coef(db, mi)
                isfinite(βb) && push!(boots, βb)
            catch
            end
        end
        ci = quantile(boots, [0.025, 0.975])
        push!(out, (lab, β0, ci[1], ci[2], r2))
    end
    return out
end

df5[!, :log_c1] = log10.(max.(df5.c1, 1e-12))
att_decay = attenuation_models(df5, :decay_rate; n_boot=F5_BOOT, seed=541)
att_c1 = attenuation_models(df5, :log_c1; n_boot=F5_BOOT, seed=542)

println("\n═══ HF attenuation (beta_HF) on decay_rate ═══")
for r in eachrow(att_decay)
    @printf("%-12s  beta_HF=%.4f  95%% CI [%.4f, %.4f]  R2=%.3f\n", r.model, r.beta_hf, r.ci_lo, r.ci_hi, r.r2)
end

println("\n═══ HF attenuation (beta_HF) on log10(C1) ═══")
for r in eachrow(att_c1)
    @printf("%-12s  beta_HF=%.4f  95%% CI [%.4f, %.4f]  R2=%.3f\n", r.model, r.beta_hf, r.ci_lo, r.ci_hi, r.r2)
end

# ── Within-group partial correlations (control speed) ─────────────────────────
partial_specs = [
    (:eta, :rank, :speed),
    (:eta, :log_eigdist, :speed),
    (:rank, :decay_rate, :speed),
    (:log_eigdist, :log_h1, :speed),
    (:log_eigdist, :log_c1, :speed),
]

function bootstrap_partial_group(df::DataFrame, g::String, x::Symbol, y::Symbol, z::Symbol;
                                 n_boot=F5_BOOT, seed=600)
    d = df[df.group .== g, [:subject, x, y, z]]
    d = d[.!isnan.(d[!, x]) .& .!isnan.(d[!, y]) .& .!isnan.(d[!, z]), :]
    if nrow(d) < 8
        return (point=NaN, ci=(NaN, NaN), n=nrow(d))
    end
    p0 = partial_corr(Float64.(d[!, x]), Float64.(d[!, y]), Float64.(d[!, z]))
    rng = MersenneTwister(seed)
    boots = Float64[]
    sizehint!(boots, n_boot)
    for _ in 1:n_boot
        idx = subject_resample_indices(d.subject; rng=rng)
        db = d[idx, :]
        pb = partial_corr(Float64.(db[!, x]), Float64.(db[!, y]), Float64.(db[!, z]))
        isfinite(pb) && push!(boots, pb)
    end
    ci = quantile(boots, [0.025, 0.975])
    return (point=p0, ci=(ci[1], ci[2]), n=nrow(d))
end

println("\n═══ Within-group partial correlations (controlling speed) ═══")
for g in GROUP_ORDER
    println("\nGroup $g")
    for (x, y, z) in partial_specs
        st = bootstrap_partial_group(df5, g, x, y, z; n_boot=F5_BOOT, seed=700 + hash((g, x, y, z)) % 10_000)
        @printf("  %-22s vs %-12s | r_partial = %6.3f  95%% CI [%6.3f, %6.3f]  n=%d\n",
                String(x), String(y), st.point, st.ci[1], st.ci[2], st.n)
    end
end

# ── Clinical linkage (stroke subjects only) ───────────────────────────────────
score_paths = [
    expanduser("~/Documents/Synology_local/Python/Gait-Signatures/data/subject_scores.csv"),
    expanduser("~/Synology/Python/Gait-Signatures/data/subject_scores.csv"),
]
score_path = findfirst(isfile, score_paths)

df_clin = DataFrame()
if isnothing(score_path)
    @warn "No clinical score CSV found in expected paths. Skipping clinical analyses."
else
    clin_raw = CSV.read(score_paths[score_path], DataFrame; header=false)
    ncol(clin_raw) < 3 && error("Expected at least 3 columns in subject score file")
    rename!(clin_raw, [:subject_raw, :berg_raw, :fugl_raw])

    df_scores = DataFrame(
        subject = strip.(String.(clin_raw.subject_raw)),
        berg = [try_parse_float(v) for v in clin_raw.berg_raw],
        fugl = [try_parse_float(v) for v in clin_raw.fugl_raw],
    )

    # Keep stroke-only subjects with available scores.
    stroke_subj = unique(df5_subj.subject[df5_subj.group .!= "AB"])
    keep_score = [(!ismissing(s)) && (s in stroke_subj) for s in df_scores.subject]
    df_scores = df_scores[keep_score, :]

    df_clin = leftjoin(df5_subj[df5_subj.group .!= "AB", :], df_scores; on=:subject)
    keep_clin = [(!ismissing(b)) && (!ismissing(f)) && isfinite(Float64(b)) && isfinite(Float64(f))
                 for (b, f) in zip(df_clin.berg, df_clin.fugl)]
    df_clin = df_clin[keep_clin, :]

    println("\n═══ Clinical merge summary ═══")
    @printf("stroke subjects in metrics: %d\n", length(stroke_subj))
    @printf("subjects with Berg/Fugl:    %d\n", nrow(df_clin))

    clinical_features = [
        :eta_int, :eta_slope,
        :rank_int, :rank_slope,
        :log_eigdist_int, :log_eigdist_slope,
        :c1_int, :c1_slope,
        :decay_rate_int, :decay_rate_slope,
        :qfactor_int, :qfactor_slope,
    ]

    function boot_corr(df::DataFrame, x::Symbol, y::Symbol; n_boot=F5_BOOT, seed=811)
        keep_xy = [(!ismissing(a)) && (!ismissing(b)) && isfinite(Float64(a)) && isfinite(Float64(b))
               for (a, b) in zip(df[!, x], df[!, y])]
        d = df[keep_xy, :]
        nrow(d) < 6 && return (point=NaN, ci=(NaN, NaN), n=nrow(d))
        point = cor(Float64.(d[!, x]), Float64.(d[!, y]))
        rng = MersenneTwister(seed)
        boots = Float64[]
        sizehint!(boots, n_boot)
        n = nrow(d)
        for _ in 1:n_boot
            idx = rand(rng, 1:n, n)
            xb = Float64.(d[idx, x])
            yb = Float64.(d[idx, y])
            (std(xb) == 0 || std(yb) == 0) && continue
            push!(boots, cor(xb, yb))
        end
        ci = quantile(boots, [0.025, 0.975])
        return (point=point, ci=(ci[1], ci[2]), n=n)
    end

    rows_clin = Vector{NamedTuple}()
    for feat in clinical_features
        st_b = boot_corr(df_clin, feat, :berg; seed=1000 + hash(feat) % 10_000)
        st_f = boot_corr(df_clin, feat, :fugl; seed=2000 + hash(feat) % 10_000)
        push!(rows_clin, (feature=String(feat), score="Berg", r=st_b.point, lo=st_b.ci[1], hi=st_b.ci[2], n=st_b.n))
        push!(rows_clin, (feature=String(feat), score="Fugl", r=st_f.point, lo=st_f.ci[1], hi=st_f.ci[2], n=st_f.n))
    end
    clinical_corr = DataFrame(rows_clin)

    println("\n═══ Clinical correlations (subject-level) ═══")
    for r in eachrow(clinical_corr)
        @printf("%-22s  %-4s  r=%6.3f  95%% CI [%6.3f, %6.3f]  n=%d\n",
                r.feature, r.score, r.r, r.lo, r.hi, r.n)
    end

    # Save clinical table for quick downstream filtering.
    CSV.write("cache/fig5_clinical_correlations.csv", clinical_corr)
end

# ── Group median resolvent curves normalized to stride frequency (panel A) ───
spec = @cached cache_path("fig5_resolvent_norm_curves.jls") begin
    println("Building normalized resolvent curves…")
    xgrid = collect(range(0.3, 5.0, length=240))  # frequency normalized by fundamental
    curves_by_group = Dict(g => Matrix{Float64}(undef, 0, length(xgrid)) for g in GROUP_ORDER)

    for i in 1:n_trials
        try
            df5.speed_matched[i] || continue
            g = df5.group[i]
            A, _, _ = get_stable_dmd_operator(data[i, :, :])
            fh, gh = resolvent_gain_serial(A; n_angles=2000)
            f0 = df5.f_stride[i]
            isnan(f0) && continue
            i0 = argmin(abs.(fh .- f0))
            g0 = max(gh[i0], 1e-12)
            x = fh ./ f0
            y = gh ./ g0

            row = [interp_linear(x, y, xq) for xq in xgrid]
            size(curves_by_group[g], 1) == 0 ?
                (curves_by_group[g] = reshape(row, 1, :)) :
                (curves_by_group[g] = vcat(curves_by_group[g], reshape(row, 1, :)))
        catch
        end
    end

    med = Dict{String, Vector{Float64}}()
    q25 = Dict{String, Vector{Float64}}()
    q75 = Dict{String, Vector{Float64}}()
    for g in GROUP_ORDER
        G = curves_by_group[g]
        if size(G, 1) == 0
            med[g] = fill(NaN, length(xgrid))
            q25[g] = fill(NaN, length(xgrid))
            q75[g] = fill(NaN, length(xgrid))
            continue
        end
        med[g] = [median(filter(!isnan, vec(G[:, j]))) for j in 1:size(G, 2)]
        q25[g] = [quantile(filter(!isnan, vec(G[:, j])), 0.25) for j in 1:size(G, 2)]
        q75[g] = [quantile(filter(!isnan, vec(G[:, j])), 0.75) for j in 1:size(G, 2)]
    end
    (x=xgrid, med=med, q25=q25, q75=q75, n=Dict(g => size(curves_by_group[g], 1) for g in GROUP_ORDER))
end

# ── Plots ─────────────────────────────────────────────────────────────────────
# A: Group median normalized resolvent curves
pA = pub_plot(; xlabel="Frequency / f_stride",
                ylabel="Gain / gain at f_stride",
                yscale=:log10,
                title="(A) Normalized resolvent breadth across groups",
                legend=:topright)
for g in GROUP_ORDER
    x = spec.x
    y = spec.med[g]
    lo = spec.q25[g]
    hi = spec.q75[g]
    ok = .!isnan.(y)
    plot!(pA, x[ok], hi[ok]; fillrange=lo[ok], fillalpha=0.15,
          fillcolor=GROUP_COLORS[g], lw=0, label="")
    plot!(pA, x[ok], y[ok]; color=GROUP_COLORS[g], lw=2.4,
          label="$g (n=$(spec.n[g]))")
end
vline!(pA, [1.0, 2.0, 3.0, 4.0, 5.0]; color=:gray70, ls=:dot, lw=1, label="")
fig5a = plot(pA; size=(1.6*PUB_W, 1.2*PUB_H))
savefig(fig5a, "figures/fig5a_resolvent_norm_curve.svg")

# B: Subject-level speed slope for eta (keep this panel; most interpretable)
function slope_strip(df_sub::DataFrame, col::Symbol, ttl::String, ylab::String)
    sp = pub_plot(; title=ttl, ylabel=ylab, xticks=(1:3, GROUP_ORDER), legend=false)
    rng = MersenneTwister(42 + hash(col) % 10_000)
    for (i, g) in enumerate(GROUP_ORDER)
        vals = filter(!isnan, Float64.(df_sub[df_sub.group .== g, col]))
        n = length(vals)
        xs = i .+ 0.22 .* (rand(rng, n) .- 0.5)
        scatter!(sp, xs, vals; color=GROUP_COLORS[g], alpha=0.70,
                 markersize=PUB_MSIZ-1, markerstrokewidth=0, label="")
        if n > 0
            q = quantile(vals, [0.25, 0.5, 0.75])
            plot!(sp, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
            plot!(sp, [i, i], [q[1], q[3]]; color=:black, lw=2, label="")
        end
    end
    return sp
end

pB1 = slope_strip(df5_subj, :eta_slope, "(B1) Speed sensitivity of eta", "d eta / d speed")
fig5b = plot(pB1; size=(1.2*PUB_W, 1.1*PUB_H))
savefig(fig5b, "figures/fig5b_eta_speed_slope.svg")

# C: HF attenuation plot with component-wise sequence
function attenuation_plot(dfA::DataFrame, dfB::DataFrame)
    x = 1:nrow(dfA)
    labs = ["group+speed", "+rank", "+eigdist", "+eta"]
    sp = pub_plot(; xlabel="Added component", ylabel="HF effect size (beta)",
                  xticks=(x, labs), xrotation=18,
                  title="(C) How HF effect changes as components are added", legend=:topright)
    for (dfm, lab, col) in ((dfA, "decay_rate", :steelblue), (dfB, "log10(C1)", :darkorange))
        y = Float64.(dfm.beta_hf)
        lo = Float64.(dfm.ci_lo)
        hi = Float64.(dfm.ci_hi)
        plot!(sp, x, y; color=col, lw=2.5, marker=:circle, label=lab)
        for i in eachindex(x)
            plot!(sp, [x[i], x[i]], [lo[i], hi[i]]; color=col, lw=1.5, label="")
        end
    end
    hline!(sp, [0.0]; color=:gray60, ls=:dash, lw=1, label="")
    return sp
end

fig5c = attenuation_plot(att_decay, att_c1)
savefig(fig5c, "figures/fig5c_hf_attenuation.svg")

# D: Clinical scatter panels (Berg-focused for main text)
fig5d_berg = pub_plot(; title="Clinical Berg panels unavailable", legend=false)
if nrow(df_clin) > 0
    function clin_scatter(df, xcol::Symbol, ycol::Symbol, ttl::String, xlab::String, ylab::String)
        sp = pub_plot(; title=ttl, xlabel=xlab, ylabel=ylab, legend=:bottomright)
        for g in ["HF", "LF"]
            d = df[df.group .== g, :]
            x = Float64.(d[!, xcol]); y = Float64.(d[!, ycol])
            scatter!(sp, x, y; color=GROUP_COLORS[g], markersize=PUB_MSIZ,
                     markerstrokewidth=0, alpha=0.80, label="$g (n=$(length(y)))")
            if length(y) >= 3 && std(x) > 0
                X = hcat(ones(length(x)), x)
                β = X \ y
                xx = range(minimum(x), maximum(x), length=50)
                plot!(sp, xx, β[1] .+ β[2] .* xx; color=GROUP_COLORS[g], lw=2, label="")
            end
        end
        if nrow(df) >= 4 && std(Float64.(df[!, xcol])) > 0 && std(Float64.(df[!, ycol])) > 0
            rall = cor(Float64.(df[!, xcol]), Float64.(df[!, ycol]))
            annotate!(sp, minimum(Float64.(df[!, xcol])), maximum(Float64.(df[!, ycol])),
                      text("r=$(round(rall; digits=2))", 8, :black, :left))
        end
        return sp
    end

    pD1 = clin_scatter(df_clin, :rank_int, :berg,
                       "(D1) Baseline rank vs Berg",
                       "Rank intercept", "Berg score")
    pD2 = clin_scatter(df_clin, :rank_slope, :berg,
                       "(D2) Rank speed-slope vs Berg",
                       "Rank slope", "Berg score")
    fig5d_berg = plot(pD1, pD2; layout=(1,2), size=(2*PUB_W, PUB_H))
    savefig(fig5d_berg, "figures/fig5d_berg_scatter.svg")
end

# E: Nonlinear relation between eig-distance and fundamental gain
pE = pub_plot(; xlabel="-log10 eig-distance", ylabel="log10 fundamental gain",
                title="(E) Fundamental gain vs eigenvalue proximity",
                legend=:bottomright)
for g in GROUP_ORDER
    d = df5[df5.group .== g, :]
    ok = .!isnan.(d.log_eigdist) .& .!isnan.(d.log_h1)
    x = .-d.log_eigdist[ok]
    y = d.log_h1[ok]
    scatter!(pE, x, y; color=GROUP_COLORS[g], alpha=0.65,
             markersize=PUB_MSIZ-1, markerstrokewidth=0,
             label="$g (n=$(length(y)))")
    if length(y) >= 3
        X = hcat(ones(length(x)), x)
        β = X \ y
        xx = range(minimum(x), maximum(x), length=50)
        yy = β[1] .+ β[2] .* xx
        plot!(pE, xx, yy; color=GROUP_COLORS[g], lw=2, label="")
    end
end
savefig(pE, "figures/fig5e_gain_vs_eigdist.svg")

# F: Gain normalized by rank (exploratory efficiency-like metric)
function trial_strip(df::DataFrame, col::Symbol, ttl::String, ylab::String)
    sp = pub_plot(; title=ttl, ylabel=ylab, xticks=(1:3, GROUP_ORDER), legend=false)
    rng = MersenneTwister(77 + hash(col) % 10_000)
    for (i, g) in enumerate(GROUP_ORDER)
        vals = filter(!isnan, Float64.(df[df.group .== g, col]))
        n = length(vals)
        xs = i .+ 0.22 .* (rand(rng, n) .- 0.5)
        scatter!(sp, xs, vals; color=GROUP_COLORS[g], alpha=0.65,
                 markersize=PUB_MSIZ-1, markerstrokewidth=0, label="")
        if n > 0
            q = quantile(vals, [0.25, 0.5, 0.75])
            plot!(sp, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
            plot!(sp, [i, i], [q[1], q[3]]; color=:black, lw=2, label="")
        end
    end
    return sp
end

pF1 = trial_strip(df5[df5.speed_matched, :], :log_h1_per_rank,
                  "(F1) Fundamental gain normalized by rank", "log10(gain_h1 / rank)")
fig5f1 = plot(pF1; size=(1.2*PUB_W, 1.1*PUB_H))
savefig(fig5f1, "figures/fig5f1_gain_per_rank.svg")

println("\nSaved: figures/fig5a_resolvent_norm_curve.svg")
println("Saved: figures/fig5b_eta_speed_slope.svg")
println("Saved: figures/fig5c_hf_attenuation.svg")
if nrow(df_clin) > 0
    println("Saved: figures/fig5d_berg_scatter.svg")
end
println("Saved: figures/fig5e_gain_vs_eigdist.svg")
println("Saved: figures/fig5f1_gain_per_rank.svg")

# Expose key objects for interactive notebook use.
fig5_trial_df = df5
fig5_subject_df = df5_subj
fig5_att_decay = att_decay
fig5_att_c1 = att_c1
fig5_clinical_df = df_clin
