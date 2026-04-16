#!/usr/bin/env julia
# Figure 1.jl — Non-normality benchmarks: simulated oscillators
#
# Panel A: Stuart-Landau, sweep μ          → Henrici vs μ
# Panel B: FitzHugh-Nagumo, sweep ε        → Henrici vs ε (timescale separation)
# Panel C: Stuart-Landau + shear, sweep c  → Henrici vs c (isochron twist)
# Panel D: Heatmap Henrici(μ, c)

include("common.jl")
using DifferentialEquations

# ── Simulators ────────────────────────────────────────────────────────────────
function simulate_stuart_landau(mu, upsilon, u0; T=30.0, dt=0.05, omega=1.0)
    function rhs!(du, u, _, _)
        x, y = u; r2 = x^2 + y^2
        du[1] = mu*x - upsilon*r2*x - omega*y
        du[2] = mu*y - upsilon*r2*y + omega*x
    end
    sol = solve(ODEProblem(rhs!, Float64[u0...], (0.0, T)),
                saveat=0.0:dt:T, reltol=1e-9, abstol=1e-11)
    return Matrix(sol)
end

function simulate_sl_shear(mu, upsilon, c, u0; omega=1.0, T=20.0, dt=0.025)
    function rhs!(du, u, _, _)
        x, y = u; r2 = x^2 + y^2
        freq = omega + c * r2
        du[1] = mu*x - upsilon*r2*x - freq*y
        du[2] = mu*y - upsilon*r2*y + freq*x
    end
    sol = solve(ODEProblem(rhs!, Float64[u0...], (0.0, T)),
                Tsit5(), saveat=0.0:dt:T, reltol=1e-8, abstol=1e-10)
    return Matrix(sol)
end

function simulate_fhn(eps, u0; a=0.7, b=0.7, I_ext=0.5, T=60.0, dt=0.1)
    function rhs!(du, u, _, _)
        v, w = u
        du[1] = v - v^3/3 - w + I_ext
        du[2] = eps * (v + a - b*w)
    end
    sol = solve(ODEProblem(rhs!, Float64[u0...], (0.0, T)),
                Tsit5(), saveat=0.0:dt:T, reltol=1e-8, abstol=1e-10)
    return Matrix(sol)
end

# ── DMD operators (data is n_vars × n_time) ───────────────────────────────────
function construct_hankel(data, tau, d)
    m, n = size(data)
    n_cols = n - (d-1)*tau
    H = zeros(m*d, n_cols)
    for i in 0:d-1
        H[i*m+1:(i+1)*m, :] = data[:, i*tau+1 : i*tau+n_cols]
    end
    return H
end

function delay_dmd_operator(data::AbstractMatrix; tau=1, d=100, svd_tol=1e-10)
    H = construct_hankel(data, tau, d)
    X = H[:, 1:end-1]; Y = H[:, 2:end]
    F = svd(X, full=false)
    r = sum(F.S .> F.S[1] * svd_tol)
    r < 2 && return nothing
    return F.U[:,1:r]' * Y * F.Vt[1:r,:]' * Diagonal(1.0 ./ F.S[1:r])
end

function delay_dmd_operator(XY::Tuple; tau=1, d=100, svd_tol=1e-10)
    X, Y = XY
    F = svd(X, full=false)
    r = sum(F.S .> F.S[1] * svd_tol)
    r < 2 && return nothing
    return F.U[:,1:r]' * Y * F.Vt[1:r,:]' * Diagonal(1.0 ./ F.S[1:r])
end

function bootstrap_henrici(samples, rng; n_boot=2000)
    n     = length(samples)
    means = [mean(samples[rand(rng, 1:n, n)]) for _ in 1:n_boot]
    return mean(means), quantile(means, 0.025), quantile(means, 0.975)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A: Stuart-Landau — sweep μ  [threaded, cached]
# ═══════════════════════════════════════════════════════════════════════════════
ups_fixed  = 1.0
mu_line    = range(0.01, 1.8, length=50)
ic_pool_sl = [(0.9,0.3),(0.6,0.1),(0.2,0.7),(1.1,-0.2),(0.05,0.05),
              (0.8,-0.4),(0.4,0.9),(0.7,-0.5),(0.3,0.2),(0.5,-0.3),(0.2,0.4),(0.6,0.8)]

sl_mean, sl_lo, sl_hi = @cached cache_path("fig1_panelA.jls") begin
    println("Panel A: Stuart-Landau sweep μ  ($(Threads.nthreads()) threads)…")
    _mean = fill(NaN, length(mu_line))
    _lo   = fill(NaN, length(mu_line))
    _hi   = fill(NaN, length(mu_line))
    Threads.@threads for k in eachindex(mu_line)
        mu       = mu_line[k]
        rng_k    = MersenneTwister(40_000 + k)
        samps    = Float64[]
        for u0 in ic_pool_sl
            A = delay_dmd_operator(simulate_stuart_landau(mu, ups_fixed, u0))
            A !== nothing && push!(samps, henrici_departure(A))
        end
        isempty(samps) && continue
        _mean[k], _lo[k], _hi[k] = bootstrap_henrici(samps, rng_k)
    end
    (_mean, _lo, _hi)
end

pA = pub_plot(collect(mu_line), sl_mean;
        ribbon    = (sl_mean .- sl_lo, sl_hi .- sl_mean),
        fillalpha = 0.20, color = :green,
        lw = PUB_LW, marker = :circle, markersize = 4,
        xlabel = "μ", ylabel = "Henrici departure",
        title  = "(A)  Stuart–Landau  (ω=1, υ=1)", legend = false)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B: FitzHugh-Nagumo — sweep ε  [threaded, cached]
# ═══════════════════════════════════════════════════════════════════════════════
τ_fhn = 1; k_fhn = 100; n_rep = 10; frac_rep = 0.3
eps_line    = range(0.1, 1.45, length=60)
ic_pool_fhn = [(0.5,0.1),(-1.5,0.5),(1.8,-0.2),(0.0,0.8),(-0.8,0.3),
               (1.2,-0.4),(0.3,0.6),(-0.5,0.2),(0.7,-0.1),(1.5,-0.5),(-1.0,0.4),(0.8,-0.3)]

fhn_mean, fhn_lo, fhn_hi = @cached cache_path("fig1_panelB.jls") begin
    println("Panel B: FHN sweep ε  ($(Threads.nthreads()) threads)…")
    _mean = fill(NaN, length(eps_line))
    _lo   = fill(NaN, length(eps_line))
    _hi   = fill(NaN, length(eps_line))
    Threads.@threads for k in eachindex(eps_line)
        eps      = eps_line[k]
        rng_k    = MersenneTwister(41_000 + k)
        trajs    = [simulate_fhn(eps, u0) for u0 in ic_pool_fhn]
        Xs       = [construct_hankel(d, τ_fhn, k_fhn)[:, 1:end-1] for d in trajs]
        Ys       = [construct_hankel(d, τ_fhn, k_fhn)[:, 2:end]   for d in trajs]
        X_cat    = hcat(Xs...); Y_cat = hcat(Ys...)
        samps    = Float64[]
        for _ in 1:n_rep
            idx = randperm(rng_k, size(X_cat,2))[1:round(Int, frac_rep*size(X_cat,2))]
            A   = delay_dmd_operator((X_cat[:,idx], Y_cat[:,idx]))
            A !== nothing && push!(samps, henrici_departure(A))
        end
        isempty(samps) && continue
        _mean[k], _lo[k], _hi[k] = bootstrap_henrici(samps, rng_k)
    end
    (_mean, _lo, _hi)
end

pB = pub_plot(collect(eps_line), fhn_mean;
        ribbon    = (fhn_mean .- fhn_lo, fhn_hi .- fhn_mean),
        fillalpha = 0.20, color = :royalblue,
        lw = PUB_LW, marker = :circle, markersize = 4,
        xlabel = "ε  (timescale separation)", ylabel = "Henrici departure",
        title  = "(B)  FitzHugh-Nagumo", legend = false)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel C: Stuart-Landau + isochron shear — sweep c  [threaded, cached]
# ═══════════════════════════════════════════════════════════════════════════════
c_line = range(0.0, 10.0, length=50)

shear_mean, shear_lo, shear_hi = @cached cache_path("fig1_panelC.jls") begin
    println("Panel C: SL + shear sweep c  ($(Threads.nthreads()) threads)…")
    _mean = fill(NaN, length(c_line))
    _lo   = fill(NaN, length(c_line))
    _hi   = fill(NaN, length(c_line))
    Threads.@threads for k in eachindex(c_line)
        c     = c_line[k]
        rng_k = MersenneTwister(43_000 + k)
        samps = Float64[]
        for u0 in ic_pool_sl
            A = delay_dmd_operator(simulate_sl_shear(1.0, ups_fixed, c, u0); tau=3, d=100)
            A !== nothing && push!(samps, henrici_departure(A))
        end
        isempty(samps) && continue
        _mean[k], _lo[k], _hi[k] = bootstrap_henrici(samps, rng_k)
    end
    (_mean, _lo, _hi)
end

pC = pub_plot(collect(c_line), shear_mean;
        ribbon    = (shear_mean .- shear_lo, shear_hi .- shear_mean),
        fillalpha = 0.20, color = :crimson,
        lw = PUB_LW, marker = :circle, markersize = 4,
        xlabel = "c  (isochron shear)", ylabel = "Henrici departure",
        title  = "(C)  Stuart–Landau + shear  (μ=1)", legend = false)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel D: Henrici heatmap over (μ, c)  [threaded on μ axis, cached]
# ═══════════════════════════════════════════════════════════════════════════════
mu_grid = range(0.01, 1.8, length=30)
c_grid  = range(0.0,  10.0, length=30)

h_map = @cached cache_path("fig1_panelD.jls") begin
    println("Panel D: Heatmap (μ, c)  ($(Threads.nthreads()) threads)…")
    _map = fill(NaN, length(mu_grid), length(c_grid))
    Threads.@threads for i in eachindex(mu_grid)
        mu = mu_grid[i]
        for (j, c) in enumerate(c_grid)
            samps = Float64[]
            for u0 in ic_pool_sl
                A = delay_dmd_operator(simulate_sl_shear(mu, ups_fixed, c, u0); tau=3, d=100)
                A !== nothing && push!(samps, henrici_departure(A))
            end
            isempty(samps) || (_map[i, j] = mean(samps))
        end
    end
    _map
end

pD = heatmap(collect(c_grid), collect(mu_grid), h_map;
        xlabel        = "c (isochron shear)", ylabel = "μ",
        colorbar_title = "Henrici departure",
        title         = "(D)  Henrici heatmap",
        color         = :viridis,
        margin        = PUB_MARGIN,
        guidefontsize  = PUB_GUIDE_FS,
        tickfontsize   = PUB_TICK_FS,
        titlefontsize  = PUB_TITLE_FS)

# ── Compose and save ──────────────────────────────────────────────────────────
fig1 = plot(pA, pB, pC, pD; layout=(2, 2), size=(2*PUB_W, 2*PUB_H))
savefig(fig1, "figures/fig1_simulated_benchmarks.svg")
println("\nSaved: figures/fig1_simulated_benchmarks.svg")
