using Random, Statistics, Distributions
using Polynomials, SpecialPolynomials
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
    st::Matrix{Float64} = model.s0*ones(npaths, 1)
    dt::Float64 = option.ttm/nsteps
    for _ in 1:nsteps
        z = Random.randn(rng, Float64, npaths, 1)
        st = st.*(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
    end

    vt = compute_option_payoff(option, st)
    v0 = discount_factor(model, option.ttm)*vt
    
    v0_avg = Statistics.mean(v0)
    v0_std = Statistics.std(v0)
    v0_low, v0_high = compute_confidence_interval(v0_avg, v0_std, npaths, alpha)
    return v0_avg, v0_low, v0_high
end

# American option MC value using Longstaff-Schwarz algorithm (least-squares Monte Carlo)
function compute_mc_option_price(model::BS, option::AmOption, npaths::Int64, nsteps::Int64, nbasisfns::Int64, alpha::Float64)::Tuple{Float64, Float64, Float64}
    rng = Xoshiro(1234);
    dt::Float64 = option.ttm/nsteps
    st::Matrix{Float64} = zeros(npaths, nsteps + 1)
    st[:, 1] = model.s0*ones(npaths, 1)

    for i in 1:nsteps
        z = Random.randn(rng, Float64, npaths, 1)
        st[:, i + 1] = st[:, i].*(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
    end

    vt = discount_factor(model, option.ttm)*compute_option_payoff(option, st[:, end])
    basis_fns = basis.(Laguerre{0}, (0:1:nbasisfns))

    for i in nsteps:-1:1
        vt_ee = discount_factor(model, i*dt)*compute_option_payoff(option, st[:, i])
        itm_idx = vt_ee .> 0
        if any(itm_idx)
            basis_fn_mat = reduce(hcat, [basis_fn.(st[itm_idx, i]) for basis_fn in basis_fns])
            b_reg = basis_fn_mat\vt[itm_idx]
            vt[itm_idx] = basis_fn_mat*b_reg
        end
    end

    vt_avg = Statistics.mean(vt)
    vt_std = Statistics.std(vt)
    vt_low, vt_high = compute_confidence_interval(vt_avg, vt_std, npaths, alpha)
    return vt_avg, vt_low, vt_high
end

# Model setup
r::Float64 = 0.05
vol::Float64 = 0.2
s0::Float64 = 90.0
bs_model::BS = BS(r, vol, s0)

# European option setup
strike::Float64 = 100.0
ttm::Float64 = 1.0 # in years
option_right::Right = call
eu_option::EuOption = EuOption(option_right, strike, ttm)

# MC settings
npaths::Int64 = 100000
nsteps::Int64 = 360
alpha::Float64 = 0.95

# Compute and print european option MC price
t = time()
mc_avg, mc_low, mc_high = compute_mc_option_price(bs_model, eu_option, npaths, nsteps, alpha)
duration = time() - t
@printf("[European Option]\n\tmc_price: %f\n\t%.2f%% ci: [%f, %f]\n\tduration: %fs\n", mc_avg, 100*alpha, mc_low, mc_high, duration)

# American option setup
am_option::AmOption = AmOption(option_right, strike, ttm)

# LSMC settings
nbasisfns::Int64 = 4

# Compute and print european option MC price
t = time()
mc_avg, mc_low, mc_high = compute_mc_option_price(bs_model, am_option, npaths, nsteps, nbasisfns, alpha)
duration = time() - t
@printf("[American Option]\n\tmc_price: %f\n\t%.2f%% ci: [%f, %f]\n\tduration: %fs\n", mc_avg, 100*alpha, mc_low, mc_high, duration)