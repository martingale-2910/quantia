using Random, Statistics, Distributions
using SpecialPolynomials
using Printf

# BS model specification
struct BS
    r::Float64
    vol::Float64
    s0::Float64
end

# Continuously compounded constant interest rate inferred discount factor
discount_factor(model::BS, t::Float64) = exp(-(model.r)*t)

# Option right
@enum Right call put

# Option style
@enum Style european american

# Option abstract type declaration
abstract type Option{style} end;

# Vanilla option wrapper
struct VanillaOption{style} <: Option{style}
    right::Right
    strike::Float64
    ttm::Float64
end

# European option declaration
EuOption = VanillaOption{european}

# American option declaration
AmOption = VanillaOption{american}

# Vanilla option payoff callback
compute_option_payoff(option::VanillaOption, s0) = return option.right == call ? max.(s0 .- option.strike, 0.0) : max.(option.strike .- s0, 0.0)

# Compute alpha-confidence interval of Monte Carlo estimate
function compute_confidence_interval(avg::Float64, std::Float64, nsamples::Int64, alpha::Float64)
    q = quantile(Normal(0.0, 1.0), 1 - (1 - alpha)/2)
    avg_low = avg - (q*std)/sqrt(nsamples)
    avg_high = avg + (q*std)/sqrt(nsamples)
    return avg_low, avg_high
end

# European option MC value
function compute_mc_option_price(model::BS, option::EuOption, npaths::Int64, nsteps::Int64, alpha::Float64)::Tuple{Float64, Float64, Float64}
    rng = Xoshiro(1234);
    sT::Matrix{Float64} = model.s0*ones(npaths, 1)
    dt::Float64 = option.ttm/nsteps
    for _ in 1:nsteps
        z = Random.randn(rng, Float64, npaths, 1)
        sT = sT.*(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
    end

    vT = compute_option_payoff(option, sT)
    v0 = discount_factor(model, option.ttm)*vT
    
    v0_avg = Statistics.mean(v0)
    v0_std = Statistics.std(v0)
    v0_low, v0_high = compute_confidence_interval(v0_avg, v0_std, npaths, alpha)
    return v0_avg, v0_low, v0_high
end

# American option MC value using Longstaff-Schwarz algorithm (least-squares Monte Carlo) with RNG state memorization
function compute_mc_option_price(model::BS, option::AmOption, npaths::Int64, nsteps::Int64, nbasisfns::Int64, alpha::Float64)::Tuple{Float64, Float64, Float64}
    rng = Xoshiro(1234);
    rng_states::Vector{Xoshiro} = Vector{Xoshiro}(undef, nsteps)
    dt::Float64 = option.ttm/nsteps
    sti = s0*ones(npaths, 1)

    for i in 1:nsteps
        rng_states[i] = copy(rng)
        z = Random.randn(rng, Float64, npaths, 1)
        sti = sti.*(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
    end

    vti = discount_factor(model, dt)*compute_option_payoff(option, sti)
    basis_fns = basis.(Legendre, (0:1:nbasisfns - 1))

    for i in nsteps:-1:1
        z = Random.randn(rng_states[i], Float64, npaths, 1)
        sti = sti./(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
        vti_ee = compute_option_payoff(option, sti)
        itm_idx = vti_ee .> 0
        if any(itm_idx)
            basis_fn_mat = reduce(hcat, [basis_fn.(sti[itm_idx]) for basis_fn in basis_fns])
            b_reg = basis_fn_mat\vti[itm_idx]
            vti[itm_idx] = basis_fn_mat*b_reg
        end
        vti = discount_factor(model, dt)*vti
    end

    v0_avg = Statistics.mean(vti)
    v0_std = Statistics.std(vti)
    v0_low, v0_high = compute_confidence_interval(v0_avg, v0_std, npaths, alpha)
    return v0_avg, v0_low, v0_high
end

# Model setup
r::Float64 = 0.05
vol::Float64 = 0.2
s0::Float64 = 90.0
bs_model::BS = BS(r, vol, s0)

# European option setup
strike::Float64 = 100.0
ttm::Float64 = 1.0 # in years
option_right::Right = put
eu_option::EuOption = EuOption(option_right, strike, ttm)

# MC settings
npaths::Int64 = 100000
nsteps::Int64 = 360
alpha::Float64 = 0.99

# Compute and print european option MC price
t = time()
mc_avg, mc_low, mc_high = compute_mc_option_price(bs_model, eu_option, npaths, nsteps, alpha)
duration = time() - t
@printf("[European Option]\n\tmc_price: %f\n\t%.2f%% ci: [%f, %f]\n\tduration: %fs\n", mc_avg, 100*alpha, mc_low, mc_high, duration)

# American option setup
am_option::AmOption = AmOption(option_right, strike, ttm)

# LSMC settings
nbasisfns::Int64 = 2

# Compute and print european option MC price
t = time()
mc_avg, mc_low, mc_high = compute_mc_option_price(bs_model, am_option, npaths, nsteps, nbasisfns, alpha)
duration = time() - t
@printf("[American Option]\n\tmc_price: %f\n\t%.2f%% ci: [%f, %f]\n\tduration: %fs\n", mc_avg, 100*alpha, mc_low, mc_high, duration)