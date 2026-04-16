#!/usr/bin/env julia
# Figure 2.jl — Human gait non-normality vs walking speed
#
# For each of the 299 trials, fit a Hankel DMD operator and compute the
# Henrici departure (same formula as Figure 1 / paper_analyses.jl).
# Plot: scatter of speed (cm/s) vs Henrici departure, coloured by group,
# with per-subject trajectory lines connecting speeds within each subject.

include("common.jl")

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

# ── Compute condition number and operator rank  [threaded, cached] ────────────
condnum_v, rank_v = @cached cache_path("fig2_condnum.jls") begin
    println("Computing condition numbers  ($(Threads.nthreads()) threads)…")
    _c = fill(NaN, n_trials)
    _r = zeros(Int, n_trials)
    Threads.@threads for i in 1:n_trials
        try
            Ã, _, r = get_stable_dmd_operator(data[i, :, :])
            _c[i] = cond(Ã)
            _r[i] = r
        catch
        end
    end
    (_c, _r)
end

# ── Assemble DataFrame ────────────────────────────────────────────────────────
using DataFrames
df2 = DataFrame(subject  = collect(subj_all),
                group    = collect(group_all),
                speed    = collect(speed_all),
                henrici  = henrici_v,
                condnum  = condnum_v,
                rank     = rank_v)

println("\n═══ Henrici departure  (median [IQR]) ═══")
@printf("%-6s  %8s  [%8s – %8s]\n", "Group", "Median", "Q25", "Q75")
println("─"^42)
for g in GROUP_ORDER
    vals = filter(!isnan, df2.henrici[df2.group .== g])
    q    = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %8.4f  [%8.4f – %8.4f]\n", g, q[2], q[1], q[3])
end

println("\n═══ Condition number κ(A) (median [IQR]) ═══")
for g in GROUP_ORDER
    vals = filter(!isnan, df2.condnum[df2.group .== g])
    q    = quantile(vals, [0.25, 0.5, 0.75])
    @printf("%-6s  %8.0f  [%8.0f – %8.0f]\n", g, q[2], q[1], q[3])
end

println("\n═══ Operator rank (median [IQR]) ═══")
for g in GROUP_ORDER
    vals = filter(>(0), df2.rank[df2.group .== g])
    q    = quantile(Float64.(vals), [0.25, 0.5, 0.75])
    @printf("%-6s  %8.0f  [%8.0f – %8.0f]\n", g, q[2], q[1], q[3])
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

# ── Condition number stats ────────────────────────────────────────────────────
fig2_cond_stats = (
    matched_diff = Dict(
        "HF-AB" => begin
            d1, d0 = boot_group(df2_sm, "HF"), boot_group(df2_sm, "AB")
            ok1, ok0 = .!isnan.(d1.condnum), .!isnan.(d0.condnum)
            hierarchical_bootstrap_diff(log.(d1.condnum[ok1]), d1.subject[ok1],
                                        log.(d0.condnum[ok0]), d0.subject[ok0];
                                        n_boot=4000, seed=151)
        end,
        "LF-AB" => begin
            d1, d0 = boot_group(df2_sm, "LF"), boot_group(df2_sm, "AB")
            ok1, ok0 = .!isnan.(d1.condnum), .!isnan.(d0.condnum)
            hierarchical_bootstrap_diff(log.(d1.condnum[ok1]), d1.subject[ok1],
                                        log.(d0.condnum[ok0]), d0.subject[ok0];
                                        n_boot=4000, seed=152)
        end,
        "HF-LF" => begin
            d1, d0 = boot_group(df2_sm, "HF"), boot_group(df2_sm, "LF")
            ok1, ok0 = .!isnan.(d1.condnum), .!isnan.(d0.condnum)
            hierarchical_bootstrap_diff(log.(d1.condnum[ok1]), d1.subject[ok1],
                                        log.(d0.condnum[ok0]), d0.subject[ok0];
                                        n_boot=4000, seed=153)
        end
    ),
)

println("\n═══ Condition number bootstrap (log scale, speed-matched) ═══")
for key in ("HF-AB", "LF-AB", "HF-LF")
    st = fig2_cond_stats.matched_diff[key]
    @printf("%-6s  Δmedian(log κ) = %.2f  95%% CI [%.2f, %.2f]  p %s\n",
            key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end

# ── Condition number strip chart (speed-matched, log scale) ───────────────────
fig2c = pub_plot(;
    xlabel  = "",
    ylabel  = "Condition number κ(A)",
    title   = "Operator conditioning  (AB ≤ $(Int(speed_cap)) cm/s)",
    xticks  = (1:3, GROUP_ORDER),
    yscale  = :log10,
    legend  = false,
    size    = (PUB_W, PUB_H))

rng_jitter2 = MersenneTwister(8)
for (i, g) in enumerate(GROUP_ORDER)
    vals = filter(!isnan, df2_sm.condnum[df2_sm.group .== g])
    n    = length(vals)
    xs   = i .+ 0.22 .* (rand(rng_jitter2, n) .- 0.5)
    scatter!(fig2c, xs, vals;
             color=GROUP_COLORS[g], markersize=PUB_MSIZ-1,
             markerstrokewidth=0, alpha=0.65, label="")
    if n > 0
        q = quantile(vals, [0.25, 0.5, 0.75])
        plot!(fig2c, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
        plot!(fig2c, [i, i], [q[1], q[3]];             color=:black, lw=2, label="")
    end
end
savefig(fig2c, "figures/fig2c_condnum_strip.svg")
println("Saved: figures/fig2c_condnum_strip.svg")
