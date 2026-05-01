#!/usr/bin/env julia
# Figure 0.jl — Schematic of the Hankel DMD gait-analysis pipeline

include("common.jl")

function boxshape(x, y, w, h)
    Shape([x - w/2, x + w/2, x + w/2, x - w/2],
          [y - h/2, y - h/2, y + h/2, y + h/2])
end

println("Loading data for Figure 0…")
data, speed_all, group_all, subj_all = load_gait_data()
trial_idx = findfirst(group_all .== "AB")
trial = data[trial_idx, :, :]
t = (0:size(trial, 1)-1) ./ FS

joint_labels = ["Left hip", "Left knee", "Left ankle",
                "Right hip", "Right knee", "Right ankle"]
joint_colors = [:steelblue, :royalblue3, :turquoise4,
                :darkorange2, :crimson, :purple3]

# ── Panel A: example time series ──────────────────────────────────────────────
n_show = min(260, size(trial, 1))
pA0 = pub_plot(; xlabel="Time (s)",
                 ylabel="Joint angle (z-scored)",
                 title="(A) Example bilateral joint-angle time series",
                 legend=:outerright)
for j in 1:size(trial, 2)
    plot!(pA0, t[1:n_show], trial[1:n_show, j];
          color=joint_colors[j], lw=2, label=joint_labels[j])
end

# ── Panel B: Hankel delay embedding ───────────────────────────────────────────
H = build_hankel_multi(trial, TAU_GAIT)
n_cols_show = min(110, size(H, 2))
yticks_pos = [3 + 6*l for l in 0:TAU_GAIT]
yticks_lab = ["t + $(l)" for l in 0:TAU_GAIT]
pB0 = heatmap(1:n_cols_show, 1:size(H, 1), H[:, 1:n_cols_show];
              xlabel="Hankel column",
              ylabel="Delay block",
              title="(B) Delay embedding into a Hankel matrix",
              color=:balance,
              yticks=(yticks_pos, yticks_lab),
              margin=PUB_MARGIN,
              guidefontsize=PUB_GUIDE_FS,
              tickfontsize=PUB_TICK_FS,
              titlefontsize=PUB_TITLE_FS,
              colorbar_title="value")
for y in 6:6:60
    hline!(pB0, [y + 0.5]; color=:white, lw=0.6, alpha=0.8, label="")
end
# ── Panel C: schematic of the fitted model ───────────────────────────────────
# We use a 1x1 coordinate system. Let's give the boxes more "breathing room."
pC0 = plot(; xlim=(0, 1), ylim=(0, 1), axis=false, ticks=false,
             title="(C) Fit reduced Hankel DMD and interpret geometry",
             margin=PUB_MARGIN)

# Box Coordinates: [center_x, center_y, width, height]
b1 = [0.18, 0.75, 0.32, 0.20]
b2 = [0.50, 0.75, 0.28, 0.20]
b3 = [0.82, 0.75, 0.32, 0.20]
b4 = [0.50, 0.30, 0.85, 0.30]

# Draw Boxes
for b in [b1, b2, b3, b4]
    plot!(pC0, boxshape(b[1], b[2], b[3], b[4]);
          seriestype=:shape, fillcolor=:white, linecolor=:black, lw=1.5, label="")
end

# Draw Connecting Arrows (Start X, End X), (Start Y, End Y)
plot!(pC0, [0.34, 0.36], [0.75, 0.75]; color=:black, lw=1.2, arrow=true, label="")
plot!(pC0, [0.64, 0.66], [0.75, 0.75]; color=:black, lw=1.2, arrow=true, label="")
plot!(pC0, [0.82, 0.82], [0.65, 0.45]; color=:black, lw=1.2, arrow=true, label="")

# Box 1 Text: Input
annotate!(pC0, b1[1], b1[2], text("Gait State\n(6-channel)", 9, :bold, :black, :center))
annotate!(pC0, b1[1], b1[2]-0.06, text("y(t) = [H, K, A]", 7, :black, :center))

# Box 2 Text: Process
annotate!(pC0, b2[1], b2[2], text("Hankel\nEmbedding", 9, :bold, :black, :center))
annotate!(pC0, b2[1], b2[2]-0.06, text("x(t) = [y(t); ...; y(t+τ)]", 7, :black, :center))

# Box 3 Text: Fit
annotate!(pC0, b3[1], b3[2], text("Reduced\nDMD Fit", 9, :bold, :black, :center))
annotate!(pC0, b3[1], b3[2]-0.06, text("z(t+1) ≈ Ã z(t)", 7, :black, :center))

# Box 4 Text: Interpretation (The big box)
annotate!(pC0, b4[1], b4[2]+0.08, text("Interpret Non-Normal Geometry", 10, :bold, :black, :center))
annotate!(pC0, b4[1], b4[2],      text("Strain (η)  |  Harmonic Gain  |  Asymmetry", 8, :black, :center))
annotate!(pC0, b4[1], b4[2]-0.08, text("Analyzes stability & energy growth mechanisms", 7, :italic, :gray30, :center))

# ── Panel D: short-horizon model prediction in joint space ───────────────────
Ared, Ured, r = get_stable_dmd_operator(trial)
z0 = Ured' * H[:, 1]
n_pred = min(150, size(H, 2))
Zhat = zeros(size(Ared, 1), n_pred)
Zhat[:, 1] = z0
for k in 2:n_pred
    Zhat[:, k] = Ared * Zhat[:, k-1]
end
Hhat = Ured * Zhat
t_pred = (0:n_pred-1) ./ FS

left_knee_actual  = vec(H[2, 1:n_pred])
left_knee_pred    = vec(Hhat[2, 1:n_pred])
right_knee_actual = vec(H[5, 1:n_pred])
right_knee_pred   = vec(Hhat[5, 1:n_pred])

pD0 = pub_plot(; xlabel="Time (s)",
                 ylabel="Delay-state angle",
                 title="(D) Example model predictions from the fitted operator",
                 legend=:outerright)
plot!(pD0, t_pred, left_knee_actual;  color=:royalblue3, lw=2.4, label="Left knee (data)")
plot!(pD0, t_pred, left_knee_pred;    color=:royalblue3, lw=2.0, ls=:dash, label="Left knee (model)")
plot!(pD0, t_pred, right_knee_actual; color=:crimson,    lw=2.4, label="Right knee (data)")
plot!(pD0, t_pred, right_knee_pred;   color=:crimson,    lw=2.0, ls=:dash, label="Right knee (model)")

fig0 = plot(pA0, pB0, pC0, pD0; layout=(2, 2), size=(2*PUB_W, 2*PUB_H))
savefig(fig0, "figures/fig0_hankel_dmd_pipeline.svg")
println("Saved: figures/fig0_hankel_dmd_pipeline.svg")
