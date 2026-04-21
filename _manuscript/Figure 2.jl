#!/usr/bin/env julia
# Figure 2.jl — Human gait non-normality vs walking speed
#
# For each of the 299 trials, fit a Hankel DMD operator and compute the
# Henrici departure (same formula as Figure 1 / paper_analyses.jl).
# Plot: scatter of speed (cm/s) vs Henrici departure, coloured by group,
# with per-subject trajectory lines connecting speeds within each subject.

include("common.jl")
using CSV

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading data…")
data, speed_all, group_all, subj_all = load_gait_data()
n_trials = size(data, 1)
println("  $(n_trials) trials  ×  $(size(data,2)) time points  ×  $(size(data,3)) joints")

# ── Compute Henrici departure  [threaded, cached] ─────────────────────────────
henrici_v = @cached cache_path("fig2_henrici.jls") begin
    println("Fitting DMD + Henrici  ($(Threads.nthreads()) threads)…")
    _h = fill(NaN, n_trials)
    Threads.@threads for i in 1:n_trials
        try
            Ã, _, _ = get_stable_dmd_operator(data[i, :, :])
            _h[i]   = henrici_departure(Ã)
        catch
        end
    end
    n_ok = sum(.!isnan.(_h))
    println("  Done.  $n_ok / $n_trials succeeded.")
    _h
end

# ── Compute operator rank  [threaded, cached] ─────────────────────────────────
rank_v = @cached cache_path("fig2_rank.jls") begin
    println("Computing operator ranks  ($(Threads.nthreads()) threads)…")
    _r = zeros(Int, n_trials)
    Threads.@threads for i in 1:n_trials
        try
            Ã, _, r = get_stable_dmd_operator(data[i, :, :])
            _r[i] = r
        catch
        end
    end
    _r
end

# ── Assemble DataFrame ────────────────────────────────────────────────────────
using DataFrames
df2 = DataFrame(subject  = collect(subj_all),
                group    = collect(group_all),
                speed    = collect(speed_all),
                henrici  = henrici_v,
                rank     = rank_v)

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

println("\n═══ Henrici departure  (median [IQR]) ═══")
@printf("%-6s  %8s  [%8s – %8s]\n", "Group", "Median", "Q25", "Q75")
println("─"^42)
for g in GROUP_ORDER
    vals = filter(!isnan, df2.henrici[df2.group .== g])
    q    = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %8.4f  [%8.4f – %8.4f]\n", g, q[2], q[1], q[3])
end

println("\n═══ Operator rank (median [IQR]) ═══")
for g in GROUP_ORDER
    vals = filter(>(0), df2.rank[df2.group .== g])
    q    = quantile(Float64.(vals), [0.25, 0.5, 0.75])
    @printf("%-6s  %8.0f  [%8.0f – %8.0f]\n", g, q[2], q[1], q[3])
end

# ── Subject-level speed sensitivity: d eta / d speed ─────────────────────────
rows = Vector{NamedTuple}()
for s in unique(df2.subject)
    d = sort(df2[df2.subject .== s, :], :speed)
    a_eta, b_eta = intercept_slope(Float64.(d.speed), Float64.(d.henrici))
    a_rank, b_rank = intercept_slope(Float64.(d.speed), Float64.(d.rank))
    push!(rows, (subject=s, group=d.group[1], eta_int=a_eta, eta_slope=b_eta,
                 rank_int=a_rank, rank_slope=b_rank, n_trials=nrow(d)))
end
df2_subj = DataFrame(rows)

println("\n═══ Subject-level eta slope (median [IQR]) ═══")
for g in GROUP_ORDER
    vals = filter(!isnan, df2_subj.eta_slope[df2_subj.group .== g])
    q = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %8.5f  [%8.5f – %8.5f]\n", g, q[2], q[1], q[3])
end

# ── Clinical merge: rank-speed slope vs Berg (stroke subjects) ───────────────
score_paths = [
    joinpath(DATA_DIR, "subject_scores.csv"),
    joinpath(@__DIR__, "data", "subject_scores.csv"),
    expanduser("~/Documents/Synology_local/Python/Gait-Signatures/data/subject_scores.csv"),
    expanduser("~/Synology/Python/Gait-Signatures/data/subject_scores.csv"),
]
score_path = findfirst(isfile, score_paths)

df2_clin = DataFrame()
clinical_rank_berg = (point=NaN, ci=(NaN, NaN), p=NaN, n=0)
if !isnothing(score_path)
    clin_raw = CSV.read(score_paths[score_path], DataFrame; header=false)
    if ncol(clin_raw) >= 3
        rename!(clin_raw, [:subject_raw, :berg_raw, :fugl_raw])
        df_scores = DataFrame(
            subject = strip.(String.(clin_raw.subject_raw)),
            berg = [try_parse_float(v) for v in clin_raw.berg_raw],
        )

        stroke_subj = unique(df2_subj.subject[df2_subj.group .!= "AB"])
        keep_score = [(!ismissing(s)) && (s in stroke_subj) for s in df_scores.subject]
        df_scores = df_scores[keep_score, :]

        df2_clin = leftjoin(df2_subj[df2_subj.group .!= "AB", :], df_scores; on=:subject)
        keep_clin = [(!ismissing(b)) && (!ismissing(r)) && isfinite(Float64(b)) && isfinite(Float64(r))
                 for (b, r) in zip(df2_clin.berg, df2_clin.rank_slope)]
        df2_clin = df2_clin[keep_clin, :]

        if nrow(df2_clin) >= 6
            clinical_rank_berg = hierarchical_bootstrap_corr(
                Float64.(df2_clin.rank_slope), Float64.(df2_clin.berg), df2_clin.subject;
                n_boot=4000, seed=181)
        end
    end
end

# ── Scatter: speed vs Henrici ──────────────────────────────────────────────────
GROUP_COLORS_LIGHT = Dict(
    g => RGBA(c.r, c.g, c.b, 0.35) for (g, c) in GROUP_COLORS
)

fig2 = pub_plot(;
    xlabel  = "Walking speed (cm/s)",
    ylabel  = "Henrici departure",
    title   = "Gait non-normality vs speed",
    legend  = :bottomright,
    size    = (PUB_W, PUB_H))

for g in GROUP_ORDER
    df_g = df2[df2.group .== g, :]
    for subj in unique(df_g.subject)
        df_s = sort(df_g[df_g.subject .== subj, :], :speed)
        ok   = .!isnan.(df_s.henrici)
        sum(ok) < 2 && continue
        plot!(fig2, df_s.speed[ok], df_s.henrici[ok];
              color=GROUP_COLORS_LIGHT[g], lw=1.2, label="")
    end
    ok = .!isnan.(df_g.henrici)
    scatter!(fig2, df_g.speed[ok], df_g.henrici[ok];
             color=GROUP_COLORS[g], markersize=PUB_MSIZ,
             markerstrokewidth=0, alpha=0.75,
             label="$g  (n=$(sum(ok)))", legend=:bottomright)
end
savefig(fig2, "figures/fig2_nonnormality_vs_speed.svg")
println("\nSaved: figures/fig2_nonnormality_vs_speed.svg")

# ── Strip chart (speed-matched) ────────────────────────────────────────────────
sm_flag, speed_cap = speed_match_flag(speed_all, group_all)
df2_sm = df2[sm_flag, :]

function boot_group(df, g)
    df[df.group .== g, :]
end

fig2_stats = (
    corr = Dict(
        g => begin
            d = boot_group(df2, g)
            hierarchical_bootstrap_corr(d.speed, d.henrici, d.subject;
                                        n_boot=4000, seed=100 + i)
        end for (i, g) in enumerate(GROUP_ORDER)
    ),
    corr_diff = Dict(
        "HF-AB" => begin
            d1, d0 = boot_group(df2, "HF"), boot_group(df2, "AB")
            hierarchical_bootstrap_corr_diff(d1.speed, d1.henrici, d1.subject,
                                             d0.speed, d0.henrici, d0.subject;
                                             n_boot=4000, seed=121)
        end,
        "LF-AB" => begin
            d1, d0 = boot_group(df2, "LF"), boot_group(df2, "AB")
            hierarchical_bootstrap_corr_diff(d1.speed, d1.henrici, d1.subject,
                                             d0.speed, d0.henrici, d0.subject;
                                             n_boot=4000, seed=122)
        end
    ),
    matched_diff = Dict(
        "HF-AB" => begin
            d1, d0 = boot_group(df2_sm, "HF"), boot_group(df2_sm, "AB")
            hierarchical_bootstrap_diff(d1.henrici, d1.subject, d0.henrici, d0.subject;
                                        n_boot=4000, seed=141)
        end,
        "LF-AB" => begin
            d1, d0 = boot_group(df2_sm, "LF"), boot_group(df2_sm, "AB")
            hierarchical_bootstrap_diff(d1.henrici, d1.subject, d0.henrici, d0.subject;
                                        n_boot=4000, seed=142)
        end,
        "HF-LF" => begin
            d1, d0 = boot_group(df2_sm, "HF"), boot_group(df2_sm, "LF")
            hierarchical_bootstrap_diff(d1.henrici, d1.subject, d0.henrici, d0.subject;
                                        n_boot=4000, seed=143)
        end
    ),
    clinical = Dict(
        "rank_slope_vs_berg" => clinical_rank_berg,
    )
)

println("\n═══ Figure 2 hierarchical bootstrap ═══")
for g in GROUP_ORDER
    st = fig2_stats.corr[g]
    @printf("%-6s  r = %.3f  95%% CI [%.3f, %.3f]  p %s\n",
            g, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end
for key in ("HF-AB", "LF-AB", "HF-LF")
    st = fig2_stats.matched_diff[key]
    @printf("%-6s  Δmedian = %.4f  95%% CI [%.4f, %.4f]  p %s\n",
            key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end
if isfinite(fig2_stats.clinical["rank_slope_vs_berg"].point)
    st = fig2_stats.clinical["rank_slope_vs_berg"]
    @printf("%-20s  r = %.3f  95%% CI [%.3f, %.3f]  p %s  n=%d\n",
            "rank_slope_vs_berg", st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p), nrow(df2_clin))
end

fig2b = pub_plot(;
    xlabel  = "",
    ylabel  = "Henrici departure",
    title   = "Non-normality by group  (AB ≤ $(Int(speed_cap)) cm/s)",
    xticks  = (1:3, GROUP_ORDER),
    legend  = false,
    size    = (PUB_W, PUB_H))

rng_jitter = MersenneTwister(7)
for (i, g) in enumerate(GROUP_ORDER)
    vals = filter(!isnan, df2_sm.henrici[df2_sm.group .== g])
    n    = length(vals)
    xs   = i .+ 0.22 .* (rand(rng_jitter, n) .- 0.5)
    scatter!(fig2b, xs, vals;
             color=GROUP_COLORS[g], markersize=PUB_MSIZ-1,
             markerstrokewidth=0, alpha=0.65, label="")
    if n > 0
        q = quantile(vals, [0.25, 0.5, 0.75])
        plot!(fig2b, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
        plot!(fig2b, [i, i], [q[1], q[3]];             color=:black, lw=2, label="")
    end
end
savefig(fig2b, "figures/fig2b_nonnormality_strip.svg")
println("Saved: figures/fig2b_nonnormality_strip.svg")

fig2c = pub_plot(;
    xlabel  = "",
    ylabel  = "d eta / d speed",
    title   = "Subject-level speed sensitivity",
    xticks  = (1:3, GROUP_ORDER),
    legend  = false,
    size    = (PUB_W, PUB_H))

hline!(fig2c, [0.0]; color=:gray65, ls=:dash, lw=1, label="")
rng_slope = MersenneTwister(17)
for (i, g) in enumerate(GROUP_ORDER)
    vals = filter(!isnan, df2_subj.eta_slope[df2_subj.group .== g])
    n    = length(vals)
    xs   = i .+ 0.22 .* (rand(rng_slope, n) .- 0.5)
    scatter!(fig2c, xs, vals;
             color=GROUP_COLORS[g], markersize=PUB_MSIZ-1,
             markerstrokewidth=0, alpha=0.70, label="")
    if n > 0
        q = quantile(vals, [0.25, 0.5, 0.75])
        plot!(fig2c, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
        plot!(fig2c, [i, i], [q[1], q[3]];             color=:black, lw=2, label="")
    end
end
savefig(fig2c, "figures/fig2c_eta_speed_slope_strip.svg")
println("Saved: figures/fig2c_eta_speed_slope_strip.svg")

fig2d_berg = pub_plot(;
    xlabel = "Rank slope (delta rank per cm/s)",
    ylabel = "Berg score",
    title  = "Clinical association (stroke subjects)",
    legend = :bottomright,
    size   = (PUB_W, PUB_H))

if nrow(df2_clin) > 0
    for g in ["HF", "LF"]
        d = df2_clin[df2_clin.group .== g, :]
        x = Float64.(d.rank_slope)
        y = Float64.(d.berg)
        scatter!(fig2d_berg, x, y; color=GROUP_COLORS[g], alpha=0.80,
                 markersize=PUB_MSIZ, markerstrokewidth=0,
                 label="$g (n=$(length(y)))")
        if length(y) >= 3 && std(x) > 0
            X = hcat(ones(length(x)), x)
            β = X \ y
            xx = range(minimum(x), maximum(x), length=50)
            plot!(fig2d_berg, xx, β[1] .+ β[2] .* xx; color=GROUP_COLORS[g], lw=2, label="")
        end
    end
    st = fig2_stats.clinical["rank_slope_vs_berg"]
    if isfinite(st.point)
        annotate!(fig2d_berg,
                  minimum(Float64.(df2_clin.rank_slope)),
                  maximum(Float64.(df2_clin.berg)),
                  text("r=$(round(st.point; digits=2)), p $(fmt_pvalue(st.p))", 8, :black, :left))
    end
else
    annotate!(fig2d_berg, 0.5, 0.5, text("Clinical scores unavailable", 9, :black))
end
savefig(fig2d_berg, "figures/fig2d_rank_slope_vs_berg.svg")
println("Saved: figures/fig2d_rank_slope_vs_berg.svg")
