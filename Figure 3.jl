#!/usr/bin/env julia
# Figure 3.jl — Harmonic resolvent gain profiles (human gait)
#
# Panel A: absolute gain at harmonics 1–5, all trials
# Panel B: normalised fall-off (gain / gain_h1), all trials
# Panel C: absolute gain faceted by speed bin (AB / HF / LF columns)
# Panel D: speed-matched comparison, normalised fall-off

include("common.jl")
using FFTW
using CSV

const N_HARMONICS = 5
const N_ANGLES    = 1000
const N_SWEEP_POINTS = div(N_ANGLES, 2) + 1

function intercept_slope(speed::AbstractVector, y::AbstractVector)
    ok = .!isnan.(speed) .& .!isnan.(y)
    x = speed[ok]
    v = y[ok]
    length(v) == 0 && return (NaN, NaN)
    length(v) == 1 && return (v[1], NaN)
    std(x) == 0 && return (mean(v), 0.0)
    x_center = x .- mean(x)
    X = hcat(ones(length(v)), x_center)
    β = X \ v
    return (β[1], β[2])
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

function harmonic_gains_at(A::AbstractMatrix, f_stride::Float64;
                            n_harmonics=N_HARMONICS, fs=FS)
    n = size(A, 1)
    map(1:n_harmonics) do h
        z = exp(im * 2π * (f_stride * h) / fs)
        1.0 / max(minimum(svdvals(z * I(n) - A)), RESOLVENT_FLOOR)
    end
end

# Single-threaded resolvent sweep for use inside the outer threaded loop
function resolvent_gain_serial(A::AbstractMatrix; n_angles=N_ANGLES, fs=FS)
    thetas = range(0, 2π, length=n_angles+1)[1:end-1]
    n      = size(A, 1)
    gains  = [1.0 / max(minimum(svdvals(exp(im*θ) * I(n) - A)), RESOLVENT_FLOOR)
              for θ in thetas]
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq     = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

const FIG2B_FREQS = collect(range(0, FS / 2, length=N_SWEEP_POINTS))
const FIG2B_FREQ_MASK = (FIG2B_FREQS .>= 0.3) .& (FIG2B_FREQS .<= 5.0)

function sweep_profile_stats(mat::AbstractMatrix)
    med = fill(NaN, size(mat, 2))
    q25 = fill(NaN, size(mat, 2))
    q75 = fill(NaN, size(mat, 2))
    for j in 1:size(mat, 2)
        vals = filter(!isnan, Float64.(mat[:, j]))
        if !isempty(vals)
            q = quantile(vals, [0.25, 0.5, 0.75])
            q25[j] = q[1]
            med[j] = q[2]
            q75[j] = q[3]
        end
    end
    return med, q25, q75
end

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading data…")
data, speed_all, group_all, subj_all = load_gait_data()
n_trials = size(data, 1)

# ── DMD + harmonic gains  [threaded, cached] ──────────────────────────────────
ranks_v, f_strides_v, hgain_v, sweep_gains_v = @cached cache_path("fig3_harmonic_v2.jls") begin
    println("Running DMD on $n_trials trials  ($(Threads.nthreads()) threads)…")
    _ranks    = zeros(Int, n_trials)
    _fstrides = fill(NaN, n_trials)
    _hgain    = fill(NaN, n_trials, N_HARMONICS)
    _sweep    = fill(NaN, n_trials, N_SWEEP_POINTS)
    Threads.@threads for i in 1:n_trials
        try
            A_s, _, r      = get_stable_dmd_operator(data[i, :, :])
            fh, gh         = resolvent_gain_serial(A_s)
            f_s            = find_stride_freq(fh, gh)
            _ranks[i]      = r
            _fstrides[i]   = f_s
            _hgain[i, :]   = harmonic_gains_at(A_s, f_s)
            if length(gh) == N_SWEEP_POINTS
                _sweep[i, :] = gh
            end
        catch
        end
    end
    n_ok = sum(.!isnan.(_fstrides))
    println("  Done.  $n_ok / $n_trials succeeded.")
    (_ranks, _fstrides, _hgain, _sweep)
end

# ── Eigenvalue–stride distance  [threaded, cached] ───────────────────────────
eig_dist_v = @cached cache_path("fig3_eig_dist.jls") begin
    println("Computing eigenvalue–stride distance…")
    _d = fill(NaN, n_trials)
    Threads.@threads for i in 1:n_trials
        try
            isnan(f_strides_v[i]) && continue
            A_s, _, _ = get_stable_dmd_operator(data[i, :, :])
            z = exp(im * 2π * f_strides_v[i] / FS)
            eigs = eigvals(A_s)
            _d[i] = minimum(abs.(eigs .- z))
        catch
        end
    end
    _d
end

# ── DataFrame ─────────────────────────────────────────────────────────────────
using DataFrames
df3 = DataFrame(subject  = collect(subj_all),
                group    = collect(group_all),
                speed    = collect(speed_all),
                rank     = ranks_v,
                f_stride = f_strides_v,
                eig_dist = eig_dist_v)
for h in 1:N_HARMONICS
    df3[!, Symbol("gain_h$h")]  = hgain_v[:, h]
    df3[!, Symbol("gnorm_h$h")] = hgain_v[:, h] ./ hgain_v[:, 1]
end
# Spectral decay rate: slope of log(normalised gain) vs harmonic number (h=1..5).
# More negative = steeper fall-off. One scalar per trial.
function spectral_decay_rate(gnorms)
    # gnorms is a length-5 vector of gain_h / gain_h1
    any(isnan, gnorms) && return NaN
    any(<=(0), gnorms) && return NaN
    y = log.(gnorms)
    x = collect(1.0:5.0)
    # simple OLS slope
    xm = mean(x); ym = mean(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end

decay_v = [spectral_decay_rate(hgain_v[i, :] ./ hgain_v[i, 1]) for i in 1:n_trials]
df3[!, :decay_rate] = decay_v

sm_flag, speed_cap = speed_match_flag(speed_all, group_all)
df3[!, :speed_matched] = sm_flag
df3[!, :group2] = make_group2(df3.group)

println("\n═══ Spectral decay rate (median [IQR]) ═══")
for g in GROUP_ORDER
    vals = filter(!isnan, df3.decay_rate[df3.group .== g])
    q = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %.3f  [%.3f – %.3f]\n", g, q[2], q[1], q[3])
end

println("\n═══ Eigenvalue–stride distance (median) ═══")
for g in GROUP_ORDER
    vals = filter(!isnan, df3.eig_dist[df3.group .== g])
    q = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %.2e  [%.2e – %.2e]\n", g, q[2], q[1], q[3])
end

raw_cols  = [Symbol("gain_h$h")  for h in 1:N_HARMONICS]
norm_cols = [Symbol("gnorm_h$h") for h in 1:N_HARMONICS]
harm_xs   = collect(1:N_HARMONICS)
xtk       = (1:N_HARMONICS, string.(1:N_HARMONICS))

function swept_profile_stats(mat::AbstractMatrix)
    med = fill(NaN, size(mat, 2))
    q25 = fill(NaN, size(mat, 2))
    q75 = fill(NaN, size(mat, 2))
    for j in 1:size(mat, 2)
        vals = filter(!isnan, Float64.(mat[:, j]))
        if !isempty(vals)
            q = quantile(vals, [0.25, 0.5, 0.75])
            q25[j] = q[1]
            med[j] = q[2]
            q75[j] = q[3]
        end
    end
    return med, q25, q75
end

function participation_ratio_from_svals(svals::AbstractVector)
    s2 = svals .^ 2
    den = sum(s2 .^ 2)
    den <= 0 && return NaN
    return (sum(s2) ^ 2) / den
end

part_ratio_v = @cached cache_path("fig3_participation_ratio.jls") begin
    println("Computing participation ratio from Hankel singular values…")
    _pr = fill(NaN, n_trials)
    Threads.@threads for i in 1:n_trials
        try
            H = build_hankel_multi(data[i, :, :], TAU_GAIT)
            X = H[:, 1:end-1]
            _pr[i] = participation_ratio_from_svals(svdvals(X))
        catch
        end
    end
    _pr
end
df3[!, :part_ratio] = part_ratio_v

rows_pr = Vector{NamedTuple}()
for s in unique(df3.subject)
    d = sort(df3[(df3.subject .== s) .& .!isnan.(df3.part_ratio), :], :speed)
    a_pr, b_pr = intercept_slope(Float64.(d.speed), Float64.(d.part_ratio))
    push!(rows_pr, (subject=s, group=nrow(d) == 0 ? "NA" : d.group[1],
                    pr_int=a_pr, pr_slope=b_pr, n_trials=nrow(d)))
end
df3_subj_pr = DataFrame(rows_pr)

score_paths = [
    joinpath(DATA_DIR, "subject_scores.csv"),
    joinpath(@__DIR__, "data", "subject_scores.csv"),
    expanduser("~/Documents/Synology_local/Python/Gait-Signatures/data/subject_scores.csv"),
    expanduser("~/Synology/Python/Gait-Signatures/data/subject_scores.csv"),
]
score_path = findfirst(isfile, score_paths)

df3_clin = DataFrame()
clinical_pr_berg = (point=NaN, ci=(NaN, NaN), p=NaN, n=0)
clinical_pr_fugl = (point=NaN, ci=(NaN, NaN), p=NaN, n=0)
if !isnothing(score_path)
    clin_raw = CSV.read(score_paths[score_path], DataFrame; header=false)
    if ncol(clin_raw) >= 3
        rename!(clin_raw, [:subject_raw, :berg_raw, :fugl_raw])
        df_scores = DataFrame(
            subject = strip.(String.(clin_raw.subject_raw)),
            berg = [try_parse_float(v) for v in clin_raw.berg_raw],
            fugl = [try_parse_float(v) for v in clin_raw.fugl_raw],
        )

        df3_clin = leftjoin(df3_subj_pr[df3_subj_pr.group .!= "AB", :], df_scores; on=:subject)

        keep_berg = [(!ismissing(b)) && isfinite(Float64(b)) && isfinite(Float64(p))
                     for (b, p) in zip(df3_clin.berg, df3_clin.pr_slope)]
        df3_clin_berg = df3_clin[keep_berg, :]
        if nrow(df3_clin_berg) >= 6
            clinical_pr_berg = hierarchical_bootstrap_corr(
                Float64.(df3_clin_berg.pr_slope), Float64.(df3_clin_berg.berg), df3_clin_berg.subject;
                n_boot=4000, seed=281)
            clinical_pr_berg = merge(clinical_pr_berg, (n=nrow(df3_clin_berg),))
        end

        keep_fugl = [(!ismissing(f)) && isfinite(Float64(f)) && isfinite(Float64(p))
                     for (f, p) in zip(df3_clin.fugl, df3_clin.pr_slope)]
        df3_clin_fugl = df3_clin[keep_fugl, :]
        if nrow(df3_clin_fugl) >= 6
            clinical_pr_fugl = hierarchical_bootstrap_corr(
                Float64.(df3_clin_fugl.pr_slope), Float64.(df3_clin_fugl.fugl), df3_clin_fugl.subject;
                n_boot=4000, seed=282)
            clinical_pr_fugl = merge(clinical_pr_fugl, (n=nrow(df3_clin_fugl),))
        end
    end
end

# ── Helpers ────────────────────────────────────────────────────────────────────
function profile_stats(df_sub, grp, cols)
    mask = df_sub.group .== grp
    med, q25, q75 = Float64[], Float64[], Float64[]
    for c in cols
        vals = filter(!isnan, Float64.(df_sub[mask, c]))
        if isempty(vals)
            push!(med, NaN); push!(q25, NaN); push!(q75, NaN)
        else
            q = quantile(vals, [0.25, 0.5, 0.75])
            push!(q25, q[1]); push!(med, q[2]); push!(q75, q[3])
        end
    end
    return med, q25, q75
end

function add_profile!(sp, xs, med, q25, q75, color; label="")
    ok = .!isnan.(med)
    any(ok) || return
    plot!(sp, xs[ok], q75[ok]; fillrange=q25[ok],
          fillalpha=0.18, fillcolor=color, lw=0, label="")
    plot!(sp, xs[ok], med[ok]; lw=PUB_LW, color=color, label=label)
end

ab_curve_bin_edges  = [0, 40, 60, 90, 130, 250]
ab_curve_bin_labels = ["<40", "40–60", "60–90", "90–130", ">130  cm/s"]

ab_curve_vals = filter(x -> isfinite(x) && x > 0, vec(sweep_gains_v[df3.group .== "AB", FIG2B_FREQ_MASK]))
ab_curve_ymin = 10.0 ^ floor(log10(minimum(ab_curve_vals)))
ab_curve_ymax = 10.0 ^ ceil(log10(maximum(ab_curve_vals)))

ab_speed_curve_panel = pub_plot(; yscale=:log10,
                      ylims=(ab_curve_ymin, ab_curve_ymax),
                      xlabel="Forcing frequency (Hz)",
                      ylabel="Resolvent gain",
                      title="(B)  AB frequency-resolved gain by speed bin",
                      legend=:topright,
                      size=(PUB_W, PUB_H),
                      guidefontsize=PUB_GUIDE_FS+1,
                      tickfontsize=PUB_TICK_FS+1,
                      legendfontsize=PUB_LEGEND_FS,
                      titlefontsize=PUB_TITLE_FS+1)

for (rk, bi) in enumerate(1:length(ab_curve_bin_labels))
    lo, hi = ab_curve_bin_edges[bi], ab_curve_bin_edges[bi+1]
    idx = (df3.group .== "AB") .& (df3.speed .>= lo) .& (df3.speed .< hi)
    mat = sweep_gains_v[idx, FIG2B_FREQ_MASK]
    med, q25, q75 = swept_profile_stats(mat)
    ok = .!isnan.(med)
    any(ok) || continue
    t = length(ab_curve_bin_labels) <= 1 ? 1.0 : (rk - 1) / (length(ab_curve_bin_labels) - 1)
    c = RGB(t, 0.0, 1.0 - t)
    plot!(ab_speed_curve_panel, FIG2B_FREQS[FIG2B_FREQ_MASK][ok], q75[ok];
        fillrange=q25[ok], fillalpha=0.14, fillcolor=c, lw=0, label="")
    plot!(ab_speed_curve_panel, FIG2B_FREQS[FIG2B_FREQ_MASK][ok], med[ok];
        lw=3.0, color=c, label=ab_curve_bin_labels[bi])
end
savefig(ab_speed_curve_panel, "figures/fig3_ab_speed_curves.svg")
println("Saved: figures/fig3_ab_speed_curves.svg")

# ── Panels A & B: all trials ───────────────────────────────────────────────────
pA = pub_plot(; yscale=:log10, xticks=xtk,
                xlabel="Harmonic  (× f_stride)", ylabel="Resolvent gain",
                title="(A)  Absolute gain — all trials", legend=:bottomleft)
pB = pub_plot(; yscale=:log10, xticks=xtk,
                xlabel="Harmonic  (× f_stride)", ylabel="Gain / gain at h=1",
                title="(B)  Normalised fall-off — all trials", legend=:bottomleft)

for g in GROUP_ORDER
    n = sum(df3.group .== g)
    med, q25, q75 = profile_stats(df3, g, raw_cols)
    add_profile!(pA, harm_xs, med, q25, q75, GROUP_COLORS[g]; label="$g (n=$n)")
    med, q25, q75 = profile_stats(df3, g, norm_cols)
    add_profile!(pB, harm_xs, med, q25, q75, GROUP_COLORS[g]; label=g)
end
hline!(pB, [1.0]; ls=:dash, color=:gray60, lw=1, label="")

# ── Panel C: speed-bin lines (absolute gain) ──────────────────────────────────
speed_bin_edges  = [0, 40, 60, 90, 130, 250]
speed_bin_labels = ["<40", "40–60", "60–90", "90–130", ">130  cm/s"]

all_raw = filter(x -> isfinite(x) && x > 0, vec(Matrix(df3[:, raw_cols])))
raw_ymin = 10.0 ^ floor(log10(minimum(all_raw)))
raw_ymax = 10.0 ^ ceil( log10(maximum(all_raw)))

speed_panels = map(GROUP_ORDER) do g
    sp = pub_plot(; yscale=:log10, ylims=(raw_ymin, raw_ymax), xticks=xtk,
                    xlabel="Harmonic",
                    ylabel=(g == "AB" ? "Resolvent gain" : ""),
                    title=g, legend=(g == "LF" ? :outerright : false),
                    size=(PUB_W3*0.8, PUB_H),
                    guidefontsize=PUB_GUIDE_FS+2,
                    tickfontsize=PUB_TICK_FS+2,
                    legendfontsize=PUB_LEGEND_FS,
                    titlefontsize=PUB_TITLE_FS+2)
    pres = [bi for bi in 1:length(speed_bin_labels)
            if sum((df3.speed .>= speed_bin_edges[bi]) .&
                   (df3.speed .<  speed_bin_edges[bi+1]) .&
                   (df3.group .== g)) > 0]
    np = length(pres)
    for (rk, bi) in enumerate(pres)
        lo, hi  = speed_bin_edges[bi], speed_bin_edges[bi+1]
        df_bin  = df3[(df3.speed .>= lo) .& (df3.speed .< hi) .& (df3.group .== g), :]
        med, _, _ = profile_stats(df_bin, g, raw_cols)
        ok = .!isnan.(med); any(ok) || continue
        t  = np <= 1 ? 1.0 : (rk-1)/(np-1)
        plot!(sp, harm_xs[ok], med[ok]; lw=3.0, color=RGB(t, 0.0, 1.0-t),
              label=speed_bin_labels[bi])
    end
    sp
end

# ── Combined Stroke speed-bin panel (HF + LF merged) ─────────────────────────
# Rename group to "Stroke" for the merged panel so profile_stats works unchanged.
df3_stroke_bins = copy(df3[df3.group .!= "AB", :])
df3_stroke_bins[!, :group] .= "Stroke"

sp_stroke_bins = pub_plot(;
    yscale=:log10, ylims=(raw_ymin, raw_ymax), xticks=xtk,
    xlabel="Harmonic",
    ylabel="Resolvent gain",
    title="Stroke",
    legend=:outerright,
    size=(PUB_W3*0.8, PUB_H),
    guidefontsize=PUB_GUIDE_FS+2,
    tickfontsize=PUB_TICK_FS+2,
    legendfontsize=PUB_LEGEND_FS,
    titlefontsize=PUB_TITLE_FS+2)

let pres = [bi for bi in 1:length(speed_bin_labels)
            if sum((df3_stroke_bins.speed .>= speed_bin_edges[bi]) .&
                   (df3_stroke_bins.speed .<  speed_bin_edges[bi+1])) > 0]
    np = length(pres)
    for (rk, bi) in enumerate(pres)
        lo, hi  = speed_bin_edges[bi], speed_bin_edges[bi+1]
        df_bin  = df3_stroke_bins[(df3_stroke_bins.speed .>= lo) .&
                                  (df3_stroke_bins.speed .<  hi), :]
        med, _, _ = profile_stats(df_bin, "Stroke", raw_cols)
        ok = .!isnan.(med); any(ok) || continue
        t  = np <= 1 ? 1.0 : (rk-1)/(np-1)
        plot!(sp_stroke_bins, harm_xs[ok], med[ok]; lw=3.0,
              color=RGB(t, 0.0, 1.0-t), label=speed_bin_labels[bi])
    end
end

# 2-panel figure: AB (section-1 reference) + combined Stroke (section-2 contrast)
fig3_ab_stroke_speed = plot(speed_panels[1], sp_stroke_bins;
                            layout=(1,2), size=(2*PUB_W3*0.8, PUB_H))
savefig(fig3_ab_stroke_speed, "figures/fig3_ab_stroke_speed_bins.svg")
println("Saved: figures/fig3_ab_stroke_speed_bins.svg")

# ── Panel D: speed-matched ─────────────────────────────────────────────────────
df3_sm = df3[df3.speed_matched, :]

function boot_group(df, g)
    df[df.group .== g, :]
end

fig3_stats = (
    matched_diff = Dict(
        "gain_h1 HF-AB" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "AB")
            hierarchical_bootstrap_diff(d1.gain_h1, d1.subject, d0.gain_h1, d0.subject;
                                        n_boot=4000, seed=201)
        end,
        "gain_h1 HF-LF" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "LF")
            hierarchical_bootstrap_diff(d1.gain_h1, d1.subject, d0.gain_h1, d0.subject;
                                        n_boot=4000, seed=202)
        end,
        "gnorm_h2 HF-AB" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "AB")
            hierarchical_bootstrap_diff(d1.gnorm_h2, d1.subject, d0.gnorm_h2, d0.subject;
                                        n_boot=4000, seed=203)
        end,
        "gnorm_h2 HF-LF" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "LF")
            hierarchical_bootstrap_diff(d1.gnorm_h2, d1.subject, d0.gnorm_h2, d0.subject;
                                        n_boot=4000, seed=204)
        end,
        "gnorm_h2 LF-AB" => begin
            d1, d0 = boot_group(df3_sm, "LF"), boot_group(df3_sm, "AB")
            hierarchical_bootstrap_diff(d1.gnorm_h2, d1.subject, d0.gnorm_h2, d0.subject;
                                        n_boot=4000, seed=205)
        end,
        "decay HF-AB" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "AB")
            ok1, ok0 = .!isnan.(d1.decay_rate), .!isnan.(d0.decay_rate)
            hierarchical_bootstrap_diff(d1.decay_rate[ok1], d1.subject[ok1],
                                        d0.decay_rate[ok0], d0.subject[ok0];
                                        n_boot=4000, seed=206)
        end,
        "decay LF-AB" => begin
            d1, d0 = boot_group(df3_sm, "LF"), boot_group(df3_sm, "AB")
            ok1, ok0 = .!isnan.(d1.decay_rate), .!isnan.(d0.decay_rate)
            hierarchical_bootstrap_diff(d1.decay_rate[ok1], d1.subject[ok1],
                                        d0.decay_rate[ok0], d0.subject[ok0];
                                        n_boot=4000, seed=207)
        end,
        "eigdist HF-AB" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "AB")
            ok1, ok0 = .!isnan.(d1.eig_dist), .!isnan.(d0.eig_dist)
            hierarchical_bootstrap_diff(d1.eig_dist[ok1], d1.subject[ok1],
                                        d0.eig_dist[ok0], d0.subject[ok0];
                                        n_boot=4000, seed=209)
        end,
        "eigdist LF-AB" => begin
            d1, d0 = boot_group(df3_sm, "LF"), boot_group(df3_sm, "AB")
            ok1, ok0 = .!isnan.(d1.eig_dist), .!isnan.(d0.eig_dist)
            hierarchical_bootstrap_diff(d1.eig_dist[ok1], d1.subject[ok1],
                                        d0.eig_dist[ok0], d0.subject[ok0];
                                        n_boot=4000, seed=210)
        end,
        "rank HF-AB" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "AB")
            ok1, ok0 = d1.rank .> 0, d0.rank .> 0
            hierarchical_bootstrap_diff(Float64.(d1.rank[ok1]), d1.subject[ok1],
                                        Float64.(d0.rank[ok0]), d0.subject[ok0];
                                        n_boot=4000, seed=211)
        end,
        "rank LF-AB" => begin
            d1, d0 = boot_group(df3_sm, "LF"), boot_group(df3_sm, "AB")
            ok1, ok0 = d1.rank .> 0, d0.rank .> 0
            hierarchical_bootstrap_diff(Float64.(d1.rank[ok1]), d1.subject[ok1],
                                        Float64.(d0.rank[ok0]), d0.subject[ok0];
                                        n_boot=4000, seed=212)
        end,
        "decay HF-LF" => begin
            d1, d0 = boot_group(df3_sm, "HF"), boot_group(df3_sm, "LF")
            ok1, ok0 = .!isnan.(d1.decay_rate), .!isnan.(d0.decay_rate)
            hierarchical_bootstrap_diff(d1.decay_rate[ok1], d1.subject[ok1],
                                        d0.decay_rate[ok0], d0.subject[ok0];
                                        n_boot=4000, seed=208)
        end
    ),
    clinical = Dict(
        "pr_slope_vs_berg" => clinical_pr_berg,
        "pr_slope_vs_fugl" => clinical_pr_fugl,
    ),
)

println("\n═══ Figure 3 hierarchical bootstrap ═══")
for key in ("gain_h1 HF-AB", "gain_h1 HF-LF", "gnorm_h2 HF-AB",
            "gnorm_h2 HF-LF", "gnorm_h2 LF-AB",
        "decay HF-AB", "decay LF-AB", "decay HF-LF",
        "eigdist HF-AB", "eigdist LF-AB", "rank HF-AB", "rank LF-AB")
    st = fig3_stats.matched_diff[key]
    @printf("%-20s  Δmedian = %.4e  95%% CI [%.4e, %.4e]  p %s\n",
            key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end
for key in ("pr_slope_vs_berg", "pr_slope_vs_fugl")
    st = fig3_stats.clinical[key]
    if isfinite(st.point)
        @printf("%-20s  r = %.3f  95%% CI [%.3f, %.3f]  p %s  n=%d\n",
                key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p), st.n)
    end
end

pD = pub_plot(; yscale=:log10, xticks=xtk,
                xlabel="Harmonic  (× f_stride)", ylabel="Gain / gain at h=1",
                title="(D)  Speed-matched  (AB ≤ $(Int(speed_cap)) cm/s)", legend=:bottomleft)
for g in GROUP_ORDER
    n = sum(df3_sm.group .== g)
    med, q25, q75 = profile_stats(df3_sm, g, norm_cols)
    add_profile!(pD, harm_xs, med, q25, q75, GROUP_COLORS[g]; label="$g (n=$n)")
end
hline!(pD, [1.0]; ls=:dash, color=:gray60, lw=1, label="")

# ── Panel E: Mechanistic interpretation (speed-matched) ─────────────────────
function strip_panel(df_plot, col::Symbol; ttl::String, ylab::String, ysc=:identity)
    sp = pub_plot(; title=ttl, ylabel=ylab, yscale=ysc,
                    xticks=(1:3, GROUP_ORDER), legend=false)
    rng = MersenneTwister(31 + hash(String(col)) % 10_000)
    for (i, g) in enumerate(GROUP_ORDER)
        vals = filter(!isnan, Float64.(df_plot[df_plot.group .== g, col]))
        n = length(vals)
        xs = i .+ 0.22 .* (rand(rng, n) .- 0.5)
        scatter!(sp, xs, vals; color=GROUP_COLORS[g], alpha=0.70,
                 markersize=PUB_MSIZ-1, markerstrokewidth=0, label="")
        if n > 0
            q = quantile(vals, [0.25, 0.5, 0.75])
            plot!(sp, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
            plot!(sp, [i, i], [q[1], q[3]];             color=:black, lw=2, label="")
        end
    end
    return sp
end

df3_mech = df3_sm[(df3_sm.rank .> 0) .& .!isnan.(df3_sm.eig_dist), :]
pE1 = strip_panel(df3_mech, :eig_dist;
                  ttl="(E1) Eigenvalue distance to stride point",
                  ylab="min |lambda - exp(i*2pi*f_stride/fs)|",
                  ysc=:log10)
pE2 = strip_panel(df3_mech, :rank;
                  ttl="(E2) Retained operator rank",
                  ylab="rank",
                  ysc=:identity)

function speed_metric_panel(df_plot, col::Symbol; ttl::String, ylab::String, ysc=:identity)
    sp = pub_plot(; title=ttl,
                    xlabel="Walking speed (cm/s)",
                    ylabel=ylab,
                    yscale=ysc,
                    legend=false)
    for g in GROUP_ORDER
        dfg = df_plot[(df_plot.group .== g) .& .!isnan.(df_plot[!, col]), :]
        isempty(dfg) && continue

        c = GROUP_COLORS[g]
        c_line = RGBA(red(c), green(c), blue(c), 0.20)

        for s in unique(dfg.subject)
            dfs = dfg[dfg.subject .== s, :]
            nrow(dfs) < 2 && continue
            ord = sortperm(dfs.speed)
            plot!(sp,
                  dfs.speed[ord],
                  Float64.(dfs[ord, col]);
                  color=c_line,
                  lw=1,
                  label="")
        end

        scatter!(sp,
                 dfg.speed,
                 Float64.(dfg[!, col]);
                 color=c,
                 alpha=0.65,
                 markersize=PUB_MSIZ-1,
                 markerstrokewidth=0,
                 label="")
    end
    return sp
end

df3_speeddim = df3[(df3.rank .> 0) .& .!isnan.(df3.part_ratio), :]
pF1 = speed_metric_panel(df3_speeddim, :rank;
                        ttl="(E1) Rank vs speed",
                        ylab="Retained operator rank")
pF2 = speed_metric_panel(df3_speeddim, :part_ratio;
                        ttl="(E2) Participation ratio vs speed",
                        ylab="Participation ratio")

pF3 = pub_plot(;
    xlabel = "d(PR) / d speed",
    ylabel = "Berg score",
    title  = "(E3) PR slope vs Berg",
    legend = :bottomright)

if nrow(df3_clin) > 0
    berg_mask = [(!ismissing(b)) && isfinite(Float64(b)) && isfinite(Float64(p))
                 for (b, p) in zip(df3_clin.berg, df3_clin.pr_slope)]
    df3_clin_berg = df3_clin[berg_mask, :]
    # Plot HF and LF points in their own colours but fit a SINGLE regression line
    # across all stroke subjects (HF and LF show the same PR-slope/Berg trend).
    for g in ["HF", "LF"]
        d = df3_clin_berg[df3_clin_berg.group .== g, :]
        isempty(d) && continue
        scatter!(pF3, Float64.(d.pr_slope), Float64.(d.berg);
                 color=GROUP_COLORS[g], alpha=0.80,
                 markersize=PUB_MSIZ, markerstrokewidth=0,
                 label="$g (n=$(nrow(d)))")
    end
    x_all = Float64.(df3_clin_berg.pr_slope)
    y_all = Float64.(df3_clin_berg.berg)
    if length(y_all) >= 3 && std(x_all) > 0
        X_mat = hcat(ones(length(x_all)), x_all)
        β = X_mat \ y_all
        xx = range(minimum(x_all), maximum(x_all), length=50)
        plot!(pF3, xx, β[1] .+ β[2] .* xx; color=:black, lw=2, ls=:dash, label="all stroke")
    end
    st = fig3_stats.clinical["pr_slope_vs_berg"]
    if isfinite(st.point)
        annotate!(pF3,
                  minimum(x_all),
                  maximum(y_all),
                  text("r=$(round(st.point; digits=2)), p $(fmt_pvalue(st.p))", 8, :black, :left))
    end
else
    annotate!(pF3, 0.5, 0.5, text("Clinical scores unavailable", 9, :black))
end

# ── Save ───────────────────────────────────────────────────────────────────────
fig3ab = plot(pA, pB; layout=(1,2), size=(2*PUB_W, PUB_H))
savefig(fig3ab, "figures/fig3ab_harmonic_gain_all.svg")

fig3c = plot(speed_panels...; layout=(1,3), size=(1.8*PUB_W3, PUB_H))
savefig(fig3c, "figures/fig3c_harmonic_speed_bins.svg")

fig3d = plot(pD; size=(1.5*PUB_W, 1.5*PUB_H))
savefig(fig3d, "figures/fig3d_harmonic_speed_matched.svg")

fig3f = plot(pF1, pF2, pF3; layout=(1,3), size=(3*PUB_W, PUB_H))
savefig(fig3f, "figures/fig3f_rank_participation_speed.svg")

println("\nSaved: figures/fig3ab, fig3c, fig3d, fig3f")

fig3_df = df3
