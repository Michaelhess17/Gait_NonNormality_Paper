#!/usr/bin/env julia
# Supplemental.jl
#
# Supplemental analyses for the gait non-normality manuscript.
# This script keeps exploratory analyses out of the main narrative while
# preserving a single entry point for running Figure 5 content.

include("common.jl")

# Silence figure-generation logging to match manuscript behavior.
using Logging
_silent(f) = redirect_stdout(devnull) do
    with_logger(NullLogger()) do
        f()
    end
end

_silent(() -> include("Figure 5.jl"))

# Figure handles exported by Figure 5.jl for supplemental use.
supplemental_figures = Dict(
    "A_resolvent_breadth" => fig5a,
    "B1_eta_speed_slope" => fig5b,
    "C_HF_attenuation" => fig5c,
    "D2_berg_rank_slope" => fig5d_berg,
    "F1_gain_per_rank" => fig5f1,
)

println("Supplemental analyses loaded from Figure 5.jl")
println("Available supplemental figure keys:")
for key in sort(collect(keys(supplemental_figures)))
    println(" - ", key)
end
