#!/usr/bin/env julia
# Figure 4.jl — Novel gait metrics and joint-coupling summary statistics
#
# Panel A: strip chart  — spatial resolvent asymmetry (max/min cross-body gain)
# Panel B: strip chart  — peak Q-factor at stride frequency
# Panel C: hip-to-knee inflow asymmetry summary
# Panel D: non-paretic-knee directional bias summary

include("common.jl")
using LsqFit

const N_JOINTS     = 6
# First 3 joints = paretic side (in stroke subjects); last 3 = non-paretic side.
# For AB subjects the split is arbitrary (left/right), labelled consistently.
const JOINT_LABELS = ["Paretic Hip",     "Paretic Knee",     "Paretic Ankle",
                      "Non-Par. Hip",    "Non-Par. Knee",    "Non-Par. Ankle"]

# ── Resolvent sweep (serial — used inside outer threaded loop) ────────────────
function resolvent_gain_serial(A::AbstractMatrix; n_angles=2000, fs=FS)
    thetas  = range(0, 2π, length=n_angles+1)[1:end-1]
    n       = size(A, 1)
    gains   = [1.0 / max(minimum(svdvals(exp(im*θ) * I(n) - A)), RESOLVENT_FLOOR)
               for θ in thetas]
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq     = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

# ── Metric functions ──────────────────────────────────────────────────────────
function spatial_asymmetry(A, U, f_stride; fs=FS, tau=TAU_GAIT, n_vars=N_JOINTS)
    n = size(A, 1)
    z = exp(im * 2π * f_stride / fs)
    R = U * inv(z * I(n) - A) * U'
    idx_1 = [j + l*n_vars for l in 0:tau for j in 1:3]
    idx_2 = [j + l*n_vars for l in 0:tau for j in 4:6]
    G12 = opnorm(R[idx_2, idx_1])
    G21 = opnorm(R[idx_1, idx_2])
    return max(G12, G21) / max(min(G12, G21), 1e-12)
end

function q_factor(A, f_stride; fs=FS)
    n      = size(A, 1)
    gf     = f -> 1.0 / max(minimum(svdvals(exp(im*2π*f/fs) * I(n) - A)), RESOLVENT_FLOOR)
    peak   = gf(f_stride)
    thresh = peak / sqrt(2)
    step   = 0.02

    fl = f_stride
    while fl > 0    && gf(fl) > thresh; fl -= step; end
    fr = f_stride
    while fr < fs/2 && gf(fr) > thresh; fr += step; end

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
    return f_stride / max((br_lo + br_hi)/2 - (bl_lo + bl_hi)/2, 1e-4)
end

function joint_resolvent_matrix(A, U, f_stride; fs=FS, tau=TAU_GAIT)
    n = size(A, 1)
    z = exp(im * 2π * f_stride / fs)
    R = U * inv(z * I(n) - A) * U'
    G = zeros(N_JOINTS, N_JOINTS)
    for jin in 1:N_JOINTS, jout in 1:N_JOINTS
        ii = [jin  + l*N_JOINTS for l in 0:tau]
        io = [jout + l*N_JOINTS for l in 0:tau]
        G[jout, jin] = opnorm(R[io, ii])
    end
    return G
end

norm_mat(G) = (m = maximum(G); m > 0 ? G ./ m : G)

function npk_inflow_asymmetry(G)
    Gn = norm_mat(G)
    npk_in = mean((Gn[5, 1], Gn[5, 4]))
    pk_in  = mean((Gn[2, 1], Gn[2, 4]))
    return log(max(npk_in, 1e-15) / max(pk_in, 1e-15))
end

function npk_directional_bias(G)
    Gn = norm_mat(G)
    vals = (
        log(max(Gn[5, 1], 1e-15) / max(Gn[1, 5], 1e-15)),
        log(max(Gn[5, 4], 1e-15) / max(Gn[4, 5], 1e-15)),
    )
    return mean(vals)
end

function global_pair_asymmetry(G)
    Gn = norm_mat(G)
    vals = Float64[]
    for a in 1:N_JOINTS, b in a+1:N_JOINTS
        push!(vals, abs(log(max(Gn[a, b], 1e-15) / max(Gn[b, a], 1e-15))))
    end
    return mean(vals)
end

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading data…")
data, speed_all, group_all, subj_all = load_gait_data()
n_trials = size(data, 1)

# ── Compute all metrics  [threaded, cached] ───────────────────────────────────
asym_v, qfac_v, jgains = @cached cache_path("fig4_metrics.jls") begin
    println("Computing metrics  ($(Threads.nthreads()) threads)…")
    _asym   = fill(NaN, n_trials)
    _qfac   = fill(NaN, n_trials)
    _jgains = fill(NaN, n_trials, N_JOINTS, N_JOINTS)
    Threads.@threads for i in 1:n_trials
        try
            A, U, _ = get_stable_dmd_operator(data[i, :, :])
            fh, gh  = resolvent_gain_serial(A)
            fs_     = find_stride_freq(fh, gh)
            _asym[i]       = spatial_asymmetry(A, U, fs_)
            _qfac[i]       = q_factor(A, fs_)
            _jgains[i,:,:] = joint_resolvent_matrix(A, U, fs_)
        catch
        end
    end
    n_ok = sum(.!isnan.(_asym))
    println("  Done.  $n_ok / $n_trials succeeded.")
    (_asym, _qfac, _jgains)
end

# ── DataFrame ─────────────────────────────────────────────────────────────────
using DataFrames

# Cross-body directional gains derived from the cached joint resolvent matrices.
# jgains[i, j_out, j_in]: opnorm of resolvent block (input joint j_in → output joint j_out).
# Paretic joints = 1:3, Non-paretic = 4:6.
# G_p2np[i]: mean gain of paretic → non-paretic blocks (how strongly paretic drives non-paretic)
# G_np2p[i]: mean gain of non-paretic → paretic blocks
G_p2np_v = [any(isnan, jgains[i,:,:]) ? NaN : mean(jgains[i, 4:6, 1:3]) for i in 1:n_trials]
G_np2p_v = [any(isnan, jgains[i,:,:]) ? NaN : mean(jgains[i, 1:3, 4:6]) for i in 1:n_trials]
npk_inflow_asym_v = [any(isnan, jgains[i,:,:]) ? NaN : npk_inflow_asymmetry(jgains[i,:,:])
                     for i in 1:n_trials]
npk_bias_v = [any(isnan, jgains[i,:,:]) ? NaN : npk_directional_bias(jgains[i,:,:])
              for i in 1:n_trials]
global_pair_asym_v = [any(isnan, jgains[i,:,:]) ? NaN : global_pair_asymmetry(jgains[i,:,:])
                      for i in 1:n_trials]

df4 = DataFrame(subject = collect(subj_all),
                group   = collect(group_all),
                speed   = collect(speed_all),
                asym    = asym_v,
                qfactor = qfac_v,
                G_p2np  = G_p2np_v,
                G_np2p  = G_np2p_v,
                dir_bias = log.(G_p2np_v ./ G_np2p_v),
                npk_inflow_asym = npk_inflow_asym_v,
                npk_bias = npk_bias_v,
                global_pair_asym = global_pair_asym_v)
sm_flag, speed_cap = speed_match_flag(speed_all, group_all)
df4[!, :speed_matched] = sm_flag
df4_sm = df4[sm_flag .& (df4.asym .> 0) .& (df4.qfactor .> 0) .&
             .!isnan.(df4.npk_inflow_asym) .& .!isnan.(df4.npk_bias) .&
             .!isnan.(df4.global_pair_asym), :]

println("\n═══ Median metrics (speed-matched) ═══")
@printf("%-6s  %10s  %10s\n", "Group", "Asymmetry", "Q-factor")
println("─"^30)
for g in GROUP_ORDER
    va = filter(!isnan, df4_sm.asym[df4_sm.group .== g])
    vq = filter(!isnan, df4_sm.qfactor[df4_sm.group .== g])
    @printf("%-6s  %10.3f  %10.3f\n", g,
            isempty(va) ? NaN : median(va),
            isempty(vq) ? NaN : median(vq))
end

# ── Strip charts (Panels A & B) ────────────────────────────────────────────────
rng4 = MersenneTwister(11)
function make_strip(df_plot, col, ylab, title_str; ysc=:identity, add_zero=false)
    sp = pub_plot(; title=title_str, ylabel=ylab, yscale=ysc,
                    xticks=(1:3, GROUP_ORDER), legend=false)
    add_zero && hline!(sp, [0.0]; color=:gray65, ls=:dash, lw=1, label="")
    for (i, g) in enumerate(GROUP_ORDER)
        vals = filter(!isnan, Float64.(df_plot[df_plot.group .== g, col]))
        n    = length(vals)
        xs   = i .+ 0.22 .* (rand(rng4, n) .- 0.5)
        scatter!(sp, xs, vals; color=GROUP_COLORS[g], alpha=0.65,
                 markersize=PUB_MSIZ-1, markerstrokewidth=0, label="")
        if n > 0
            q = quantile(vals, [0.25, 0.5, 0.75])
            plot!(sp, [i-0.22, i+0.22], [q[2], q[2]]; color=:black, lw=3, label="")
            plot!(sp, [i, i], [q[1], q[3]];             color=:black, lw=2, label="")
        end
    end
    return sp
end

pA4 = make_strip(df4_sm, :asym,    "Max/Min cross-body gain",        "(A)  Spatial asymmetry")
pB4 = make_strip(df4_sm, :qfactor, "f_stride / FWHM",                "(B)  Peak Q-factor"; ysc=:log10)
pC4 = make_strip(df4_sm, :G_p2np,  "Mean resolvent gain",             "(C)  Paretic → Non-paretic")
pD4 = make_strip(df4_sm, :G_np2p,  "Mean resolvent gain",             "(D)  Non-paretic → Paretic")

fig4ab = plot(pA4, pB4, pC4, pD4; layout=(2,2), size=(2*PUB_W, 2*PUB_H))
savefig(fig4ab, "figures/fig4ab_novel_metrics_strips.svg")
println("\nSaved: figures/fig4ab_novel_metrics_strips.svg")

# ── Joint-summary strips (replace matrix heatmaps) ────────────────────────────
pE4 = make_strip(df4_sm, :npk_inflow_asym,
                 "log(hip→NP knee / hip→P knee)",
                 "(E)  Knee-target asymmetry"; add_zero=true)
pF4 = make_strip(df4_sm, :npk_bias,
                 "Mean log[(hip→NP knee)/(NP knee→hip)]",
                 "(F)  Non-paretic-knee directional bias"; add_zero=true)
pG4 = make_strip(df4_sm, :global_pair_asym,
                 "Mean |log(G[i→j] / G[j→i])|",
                 "(G)  Global pairwise asymmetry"; add_zero=false)

fig4efg = plot(pE4, pF4, pG4; layout=(1,3), size=(3*PUB_W, PUB_H))
savefig(fig4efg, "figures/fig4efg_joint_summary_metrics.svg")
println("Saved: figures/fig4efg_joint_summary_metrics.svg")

function boot_group(df, g)
    df[df.group .== g, :]
end

fig4_stats = (
    matched_diff = Dict(
        "asym HF-AB" => begin
            d1, d0 = boot_group(df4_sm, "HF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.asym, d1.subject, d0.asym, d0.subject;
                                        n_boot=4000, seed=301)
        end,
        "asym LF-AB" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.asym, d1.subject, d0.asym, d0.subject;
                                        n_boot=4000, seed=302)
        end,
        "asym LF-HF" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "HF")
            hierarchical_bootstrap_diff(d1.asym, d1.subject, d0.asym, d0.subject;
                                        n_boot=4000, seed=303)
        end,
        "qfactor HF-AB" => begin
            d1, d0 = boot_group(df4_sm, "HF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.qfactor, d1.subject, d0.qfactor, d0.subject;
                                        n_boot=4000, seed=304)
        end,
        "qfactor HF-LF" => begin
            d1, d0 = boot_group(df4_sm, "HF"), boot_group(df4_sm, "LF")
            hierarchical_bootstrap_diff(d1.qfactor, d1.subject, d0.qfactor, d0.subject;
                                        n_boot=4000, seed=305)
        end,
        "qfactor LF-AB" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.qfactor, d1.subject, d0.qfactor, d0.subject;
                                        n_boot=4000, seed=306)
        end,
        "npk_inflow_asym HF-AB" => begin
            d1, d0 = boot_group(df4_sm, "HF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.npk_inflow_asym, d1.subject,
                                        d0.npk_inflow_asym, d0.subject;
                                        n_boot=4000, seed=307)
        end,
        "npk_inflow_asym LF-AB" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.npk_inflow_asym, d1.subject,
                                        d0.npk_inflow_asym, d0.subject;
                                        n_boot=4000, seed=308)
        end,
        "npk_bias LF-AB" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.npk_bias, d1.subject, d0.npk_bias, d0.subject;
                                        n_boot=4000, seed=309)
        end,
        "global_pair_asym HF-AB" => begin
            d1, d0 = boot_group(df4_sm, "HF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.global_pair_asym, d1.subject,
                                        d0.global_pair_asym, d0.subject;
                                        n_boot=4000, seed=310)
        end,
        "global_pair_asym LF-AB" => begin
            d1, d0 = boot_group(df4_sm, "LF"), boot_group(df4_sm, "AB")
            hierarchical_bootstrap_diff(d1.global_pair_asym, d1.subject,
                                        d0.global_pair_asym, d0.subject;
                                        n_boot=4000, seed=311)
        end
    ),
    within_group = Dict(
        "dir_bias HF" => begin
            d = boot_group(df4_sm, "HF")
            hierarchical_bootstrap_one(d.dir_bias, d.subject; statfun=mean,
                                       n_boot=4000, seed=331)
        end,
        "dir_bias LF" => begin
            d = boot_group(df4_sm, "LF")
            hierarchical_bootstrap_one(d.dir_bias, d.subject; statfun=mean,
                                       n_boot=4000, seed=332)
        end,
        "npk_bias HF" => begin
            d = boot_group(df4_sm, "HF")
            hierarchical_bootstrap_one(d.npk_bias, d.subject; statfun=mean,
                                       n_boot=4000, seed=333)
        end,
        "npk_bias LF" => begin
            d = boot_group(df4_sm, "LF")
            hierarchical_bootstrap_one(d.npk_bias, d.subject; statfun=mean,
                                       n_boot=4000, seed=334)
        end
    )
)

println("\n═══ Figure 4 hierarchical bootstrap ═══")
for key in ("asym HF-AB", "asym LF-AB", "asym LF-HF",
            "qfactor HF-AB", "qfactor HF-LF", "qfactor LF-AB",
            "npk_inflow_asym HF-AB", "npk_inflow_asym LF-AB",
            "npk_bias LF-AB", "global_pair_asym HF-AB", "global_pair_asym LF-AB")
    st = fig4_stats.matched_diff[key]
    @printf("%-24s  Δmedian = %.4f  95%% CI [%.4f, %.4f]  p %s\n",
            key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end
for key in ("dir_bias HF", "dir_bias LF", "npk_bias HF", "npk_bias LF")
    st = fig4_stats.within_group[key]
    @printf("%-24s  mean = %.4f  95%% CI [%.4f, %.4f]  p %s\n",
            key, st.point, st.ci[1], st.ci[2], fmt_pvalue(st.p))
end
