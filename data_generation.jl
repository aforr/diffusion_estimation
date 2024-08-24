# Functions for generating simulation data
include("dependencies.jl")

##########################
# Potential functions
##########################
function quadratic_potential(x::Vector)
    return (norm(x)^2)/2
end

function linear_potential(x::Vector, scale = 1)
    return -scale*sum(x)
end

function bifurcation_potential(x::Vector)
    return -x[1] - 0.5*x[1]*x[2]^2 + 0.25*x[2]^4
end

##########################
# Velocity field functions
##########################


function linear_velocity(x, A)
    v = A*x
    @assert length(v) == length(x)
    return v
end

function isotropic_sigma(D::Number)
    return x -> sqrt(2D)*I
end


##########################
# Individual samples from potential or velocity
##########################

function sample_x(n, d, mu = 0, Σ=I)
    if mu == 0
        mu = zeros(d)
    end
    sample_pdf = MvNormal(mu , Σ)
    return rand(sample_pdf, n)
end

function get_dx_from_potential(x, H::Function, D::Number, dt::Number)
    v = x -> -ForwardDiff.gradient(H, x)
    return get_dx_from_v(x, v, D, dt)
end

function get_dx_from_v(x, v::Function, D::Number, dt::Number)
    s = x -> sqrt(2D)*I
    return get_dx_from_v_sigma(x, v, s, dt)
end

function get_dx_from_v_sigma(x, v::Function, sigma::Function, dt::Number)
    x_dim = length(x)
    s = sigma(x)
    if isa(s, UniformScaling{T} where T)
        b_dim = x_dim
    else
        b_dim = size(s)[2]
    end
    return v(x).*dt + sqrt(dt).*s*randn(b_dim)
end


##########################
# Population samples from potential or velocity
##########################

function get_full_samples(n::Integer; D = 1, dt = 0.01, dims = 2, potential = quadratic_potential, sampler = sample_x)
    return get_full_samples(n, potential, D, dt, dims, sampler)
end
function get_full_samples(n::Integer, potential::Function, D::Number, dt::Number, dims::Number, sampler::Function)
    xs = sampler(n, dims)
    dxs = hcat([get_dx_from_potential(x, potential, D, dt) for x in eachcol(xs)]...)
    return xs, dxs
end

function get_samples_from_v_sigma(n::Integer, v::Function, sigma::Function, dt::Number, dims::Number, sampler::Function)
    xs = sampler(n, dims)
    dxs = hcat([get_dx_from_v_sigma(x, v, sigma, dt) for x in eachcol(xs)]...)
    return xs, dxs
end



##########################
# Pushing population forward in time
##########################

function evolve_population(xs, v::Function, sigma::Function, timestep::Number, num_steps::Number,)

    for i in 1:num_steps
        dxs = hcat([get_dx_from_v_sigma(x, v, sigma, timestep) for x in eachcol(xs)]...)
        xs = xs + dxs
    end
    return xs
end