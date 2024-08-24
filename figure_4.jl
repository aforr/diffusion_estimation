include("dependencies.jl")
include("estimation_via_covariance.jl")

figure_path = "Figures/Figure_4/"
if !isdir(figure_path)
	mkpath(figure_path)
end



tick_font_size = 14
legend_font_size = 14
label_font_size = 18
title_font_size = 18
line_width = 3
marker_size = 7

plot_size = (600, 450)


# Generate data from simulated bifurcation flow with zero diffusion
Random.seed!(100)
true_D = 0
n = 25
dims = 2
dt_time_series = 0.1
initial_covariance = diagm([3,0.5])
initial_distribution = MixtureModel([MvNormal([1,1], 0.1I), MvNormal([1,-1], 0.1I)], [0.5, 0.5])
initial_sampler = (n, d) -> rand(initial_distribution, n)

@time single_D_estimate, pop_t0, pop_t1 = run_trial(true_D,
                                    n,
                                    dims,
                                    neighbor_average = false,
                                    potential = bifurcation_potential,
                                    initial_sampler = initial_sampler,
                                    velocity_variance_scale = 0,
                                    dt_time_series = dt_time_series,
                                    return_samples = true)

estimated_relative_timescale = estimate_timescale(pop_t0, pop_t1, dt_time_series)

velocity_prefactor = estimated_relative_timescale*dt_time_series


# Part a): scatter plot of data
scatter(pop_t0[1][1,:], pop_t0[1][2,:], label = L"$t$",
 xlabel = "\$x_1\$",
 ylabel = "\$x_2\$",
 marker = marker_size,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 yticks = -1:1,
 xticks = 0.5:0.5:1.5,
 size = plot_size,
 title = L"$\textrm{a})$",
 titleloc = :left,
 titlefont = title_font_size,)
    
scatter!(pop_t1[1][1,:],
    pop_t1[1][2,:],
    marker = marker_size,
    label = L"$t'$")

quiver!(pop_t0[1][1,:],
 pop_t0[1][2,:], 
 quiver = (velocity_prefactor*pop_t0[2][1,:], velocity_prefactor*pop_t0[2][2,:]),
 linewidth = line_width,
)

savefig(figure_path*"bifurcation_samples.pdf")





# Part b): plot distribution of D estimates

n = 500;
num_trials = 100


@time D_estimates = [run_trial(true_D,
                                    n,
                                    dims,
                                    neighbor_average = false,
                                    potential = bifurcation_potential,
                                    initial_sampler = initial_sampler,
                                    velocity_variance_scale = 0,
                                    estimators = [estimate_D, estimate_D_wot],
                                    dt_time_series = dt_time_series,
                        ) for i in 1:num_trials];


D_estimates = hcat(D_estimates...)



histogram(D_estimates',
        normalize=:pdf,
        legend=:topleft, 
        labels = [L"$\widehat\mathcal{D}$"  L"$\mathcal{D}_{WOT}$"],
        xlabel = L"\mathcal{D}",
        ylabel = L"p(\mathcal{D})",
        tickfont = tick_font_size,
        guidefont = label_font_size,
        legendfont = legend_font_size,
        #yticks = -1:3,
        #xticks = -1:3,
        size = plot_size,
        title = L"$\textrm{b})$",
        titleloc = :left,
        titlefont = title_font_size,
        )

plot!([0, 0], [0, 40],
    label = L"$\textrm{True }\,\mathcal{D}$",
    linewidth = line_width,
)


savefig(figure_path*"bifurcation_D_histogram.pdf")








# Generating data for (c) and (d)
n = 500

@time fate_prob_D_estimate, pop_t0, pop_t1 = run_trial(true_D,
                                    n,
                                    dims,
                                    neighbor_average = false,
                                    potential = bifurcation_potential,
                                    initial_sampler = initial_sampler,
                                    velocity_variance_scale = 0,
                                    dt_time_series = dt_time_series,
                                    estimators = [estimate_D, estimate_D_wot],
                                    return_samples = true)


# Part c) Plot fate probabilities with WOT D estimate

# Compute OT coupling with WOT D estimate
distances = pairwise(Euclidean(),pop_t0[1], pop_t1[1], dims = 2)
state_indicator = pop_t1[1][2,:] .> 0
early_state_indicator = pop_t0[1][2,:] .> 0

ot_coupling_WOT = sinkhorn(ones(n)/n, ones(n)/n, distances, fate_prob_D_estimate[2])
WOT_state_1_prob = sum(ot_coupling_WOT[:, state_indicator], dims = 2) ./ sum(ot_coupling_WOT, dims = 2)


histogram(WOT_state_1_prob[early_state_indicator],
        alpha = 0.5,
        normalize = :pdf,
        label = L"$X_2(t)> 0$",
        xlabel = L"$P(X_2(t')>0)$",
        ylabel = "Density of cells",
        tickfont = tick_font_size,
        guidefont = label_font_size,
        legendfont = legend_font_size,
        size = plot_size,
        title = L"$\textrm{c})\qquad \mathcal{D}_{WOT}$",
        titleloc = :left,
        titlefont = title_font_size,
        )
histogram!(WOT_state_1_prob[.!early_state_indicator],
        alpha = 0.5, 
        normalize = :pdf,
        label = L"$X_2(t)< 0$",
        )
savefig(figure_path*"bifurcation_WOT_D_probs_histogram.pdf")



# Part d) plot fate probabilities with covariance D estimate

# Compute OT coupling with covariance D estimate
ot_coupling_covariance = sinkhorn(ones(n)/n, ones(n)/n, distances, fate_prob_D_estimate[1])
covariance_state_1_prob = sum(ot_coupling_covariance[:, state_indicator], dims = 2) ./ sum(ot_coupling_covariance, dims = 2)


histogram(covariance_state_1_prob[early_state_indicator],
        alpha = 0.5,
        normalize = :pdf,
        label = L"$X_2(t)> 0$",
        xlabel = L"$P(X_2(t')>0)$",
        ylabel = "Density of cells",
        tickfont = tick_font_size,
        guidefont = label_font_size,
        legendfont = legend_font_size,
        size = plot_size,
        title = L"$\textrm{d})\qquad \widehat{\mathcal{D}}$",
        titleloc = :left,
        titlefont = title_font_size,
        ylim = [0, n/5],
        )
histogram!(covariance_state_1_prob[.!early_state_indicator],
        alpha = 0.5, 
        normalize = :pdf,
        label = L"$X_2(t)< 0$",
        )
savefig(figure_path*"bifurcation_covariance_D_probs_histogram.pdf")
