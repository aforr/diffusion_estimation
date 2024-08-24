# Estimating the diffusion matrix

include("data_generation.jl")
include("utils.jl")





function estimate_timescale(pop_t0, pop_t1, dt_time_series)
	# Estimates the relative timescale of the velocity measurements
	# compared to the time difference t1-t0 between measurements

	t0_mean = mean(pop_t0[1], dims = 2)
	t0_drift_mean = mean(pop_t0[2], dims = 2)
	t1_mean = mean(pop_t1[1], dims = 2)
	relative_timescale = (norm(t0_drift_mean)^2)\dot(t0_drift_mean, t1_mean - t0_mean)/ dt_time_series

	return relative_timescale
end

function estimate_D(pop_t0, pop_t1, velocity_estimate_t0, dt_time_series)
	# Estimates a scalar diffusion coefficient from the population samples
	cov_t1 = cov(pop_t1[1]')
	relative_timescale = estimate_timescale(pop_t0, pop_t1, dt_time_series)

	num_dimensions, num_samples = size(pop_t0[1])
	
	cov_t0_pushforward = cov((pop_t0[1] + velocity_estimate_t0*relative_timescale*dt_time_series)')

	return tr((cov_t1 - cov_t0_pushforward) / (2*dt_time_series))/num_dimensions
end


function wot_D_dt(pop_t0, pop_t1)
    distances = pairwise(Euclidean(),pop_t0[1], pop_t1[1], dims = 2)
    return 0.05*median(distances)
end
function estimate_D_wot(pop_t0, pop_t1, velocity_estimate_t0, dt_time_series)
    return wot_D_dt(pop_t0, pop_t1) / dt_time_series
end



function run_trial(true_D, n, dims; 
	flow_type = :potential,
	linear_velocity_matrix = 0, # matrix A in v(x) = A*x if flow_type = :linear
	potential = x -> (norm(x) - 1)^2, # potential U in v(x) = ∇U(x) if flow_type = :potential
	neighbor_average = false, 
	velocity_variance_scale = 0,
	dt_time_series = 0.2, # time between t0 and t1
	integration_dt = 0.01, # timestep for Euler integration between t0 and t1
	min_integration_timesteps = 10, # minimum number of steps for Euler integration
	initial_sampler = (n, d) -> sample_x(n, d, ones(dims)), # initial condition normally distributed with nonzero mean
	return_samples = false,
	estimators = [estimate_D],
	)


	if flow_type == :potential
		v = x -> -ForwardDiff.gradient(potential, x)
	elseif flow_type == :linear
		v = x -> - linear_velocity_matrix*x
	else
		throw(ArgumentError("flow_type "*String(flow_type)*" not implemented."))
	end

	dt_v = dt_time_series/2 # timescale of velocity data

	pop_t0 = get_samples_from_v_sigma(n,
		v,
		isotropic_sigma(velocity_variance_scale*true_D),
		dt_v,
		dims,
		initial_sampler)


	if flow_type == :linear
		# no need to simulate trajectories
		initial_Σ = I
		D_matrix = true_D*I

		exp_At = exp(-linear_velocity_matrix*dt_time_series)
		B_term_1 = linear_velocity_matrix*exp_At*initial_Σ*exp_At' 
		B = B_term_1 + B_term_1' + 2*D_matrix - 2*exp_At *D_matrix*exp_At'
		Σ_t = solve_transpose_system(linear_velocity_matrix, B)
		@assert(Σ_t ≈ Σ_t')

		# Numerical errors make Σ_t slightly non-Hermitian, which
		# would cause an error in MvNormal()
		Σ_t = (Σ_t + Σ_t')/2

		initial_μ = ones(dims)
		μ_t = exp_At*initial_μ

		xs_t1 = rand(MvNormal(μ_t, Σ_t), n)
	else
		second_pop_t0 = get_samples_from_v_sigma(n,
			v,
			isotropic_sigma(true_D),
			dt_time_series,
			dims,
			initial_sampler)

		integration_steps = Int64(round(dt_time_series / integration_dt))
		if integration_steps < min_integration_timesteps
			integration_steps = min_integration_timesteps
			integration_dt = dt_time_series / integration_steps
		end
		xs_t1 = evolve_population(second_pop_t0[1],
			v,
			isotropic_sigma(true_D),
			integration_dt,
			integration_steps
			)

	end
	pop_t1 = (xs_t1,) # not using velocities at t1

	if neighbor_average
		num_neighbors = Int64(floor(sqrt(n)))
		velocity_estimate_t0 = hcat(estimate_vs(pop_t0[2], get_neighbors(pop_t0[1], num_neighbors)[1])...)
	else
		num_neighbors = 1
		velocity_estimate_t0 = pop_t0[2]
	end

	D_hats = [estimator(pop_t0, pop_t1, velocity_estimate_t0, dt_time_series) for estimator in estimators]
	if return_samples
		return D_hats, pop_t0, pop_t1
	else
		return D_hats
	end
end


