using Random, Statistics, Distributions
using Printf

# BS model specification
struct BS
    r::Float64
    vol::Float64
    s0::Float64
end

# Option right
@enum Right call put

# European option specification
struct EuOption
    right::Right
    strike::Float64
    ttm::Float64
end

# Vanilla option payoff callback
function compute_option_payoff(right::Right, s0, k)
    return right == call ? max.(s0 .- k, 0.0) : max.(k .- s0, 0.0)
end

function compute_confidence_interval(avg::Float64, std::Float64, nsamples::Int64, alpha::Float64)
    q = quantile(Normal(0.0, 1.0), 1 - (1 - alpha)/2)
    avg_low = avg - (q*std)/sqrt(nsamples)
    avg_high = avg + (q*std)/sqrt(nsamples)
    return avg_low, avg_high
end

# MC estimate computation callback
function compute_mc_option_price(model::BS, option::EuOption, npaths::Int64, nsteps::Int64, alpha::Float64)::Tuple{Float64, Float64, Float64}
    rng = MersenneTwister(1234);
    st::Matrix{Float64} = model.s0*ones(npaths, 1)
    dt::Float64 = option.ttm/nsteps
    for i in 1:nsteps
        z = Random.randn(rng, Float64, npaths, 1)
        st = st.*(1.0 .+ model.r*dt .+ model.vol*sqrt(dt).*z)
    end

    vt = compute_option_payoff(option.right, st, option.strike)
    v0 = exp(-model.r*option.ttm).*vt
    v0_avg = Statistics.mean(v0)
    v0_std = Statistics.std(v0)
    v0_low, v0_high = compute_confidence_interval(v0_avg, v0_std, npaths, alpha)
    return v0_avg, v0_low, v0_high
end

# Model setup
r::Float64 = 0.05
vol::Float64 = 0.2
s0::Float64 = 90.0
bs_model::BS = BS(r, vol, s0)

# Option setup
strike::Float64 = 100.0
ttm::Float64 = 1.0 # in years
option_right::Right = call
eu_option::EuOption = EuOption(option_right, strike, ttm)

# MC settings
npaths::Int64 = 1000000
nsteps::Int64 = 360
alpha::Float64 = 0.95

# Compute and print results
t = time()
mc_avg, mc_low, mc_high = compute_mc_option_price(bs_model, eu_option, npaths, nsteps, alpha)
duration = time() - t
@printf("mc_price := %f\n%.2f%% ci = [%f, %f]\nduration := %fs\n", mc_avg, 100*alpha, mc_low, mc_high, duration)