include("dependencies.jl")
include("estimation_via_covariance.jl")

figure_path = "Figures/Figure_3/"
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


Random.seed!(100)
true_D = 1
n = 25
dims = 10
num_trials = 2
dt_time_series = 0.2
linear_velocity_matrix = randn(dims, dims)

run_WOT = true
if run_WOT
    estimators = [estimate_D, estimate_D_wot]
else
    estimators = [estimate_D]
end
num_estimators = length(estimators)

save_pdfs = true



single_D_estimate, pop_t0, pop_t1 = run_trial(true_D,
                                    n,
                                    dims,
                                    flow_type = :linear,
                                    linear_velocity_matrix = linear_velocity_matrix,
                                    neighbor_average = false,
                                    velocity_variance_scale = 0,
                                    dt_time_series = dt_time_series,
                                    return_samples = true)

estimated_relative_timescale = estimate_timescale(pop_t0, pop_t1, dt_time_series)

velocity_prefactor = estimated_relative_timescale*dt_time_series





# Part a): scatter plot of data
scatter(pop_t0[1][1,:], pop_t0[1][2,:], label = L"$t_1$",
 xlabel = "\$x_1\$",
 ylabel = "\$x_2\$",
 marker = marker_size,
 shape = :diamond,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 yticks = -1:5,
 xticks = -1:3,
 size = plot_size,
 title = L"$\textrm{a})$",
 titleloc = :left,
 titlefont = title_font_size,)
    
scatter!(pop_t1[1][1,:],
    pop_t1[1][2,:],
    marker = marker_size,
    label = L"$t_2$")

quiver!(pop_t0[1][1,:],
 pop_t0[1][2,:], 
 quiver = (velocity_prefactor*pop_t0[2][1,:], velocity_prefactor*pop_t0[2][2,:]),
 linewidth = line_width,
)

if save_pdfs
    savefig(figure_path*"hat_data.pdf")
end






# Part b): MSE as a function of n with dt fixed



ns = 2 .^ (5:18);
ns_WOT = 2 .^ (5:12)


@time D_estimates_fixed_dt = [run_trial(true_D,
        n,
        dims,
        flow_type = :linear,
        linear_velocity_matrix = linear_velocity_matrix,
        neighbor_average = false,
        velocity_variance_scale = 0,
        estimators = [estimate_D],
        dt_time_series = dt_time_series,
        ) for i in 1:num_trials, n in ns];

D_estimates_fixed_dt = [D_estimates_fixed_dt[i,j][1] for i=1:num_trials, j=1:length(ns)];


MSEs_fixed_dt = dropdims(mean((D_estimates_fixed_dt .- true_D).^2, dims = 1), dims = 1)
std_errs_fixed_dt = dropdims(std((D_estimates_fixed_dt .- true_D).^2, dims = 1), dims = 1) ./ sqrt(num_trials)


scatter(ns, MSEs_fixed_dt[:, 1],
    yerr = std_errs_fixed_dt,
    label = L"$\textrm{MSE(\widehat\mathcal{D})}$",
    xscale = :log10,
    yscale = :log10,
    xlabel = "\$n\$",
     ylabel = L"\textrm{MSE}",
 marker = marker_size,
 shape = :diamond,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 size = plot_size,
 title = L"$\textrm{b})$",
 titleloc = :left,
 titlefont = title_font_size,
)

if run_WOT
    # WOT's method is much slower because it requires computing all pairwise distances
    @time D_estimates_fixed_dt_WOT = [run_trial(true_D,
            n,
            dims,
            flow_type = :linear,
            linear_velocity_matrix = linear_velocity_matrix,
            neighbor_average = false,
            velocity_variance_scale = 0,
            estimators = [estimate_D_wot],
            dt_time_series = dt_time_series,
            ) for i in 1:num_trials, n in ns_WOT];
    D_estimates_fixed_dt_WOT = [D_estimates_fixed_dt_WOT[i,j][k] for i=1:num_trials, j=1:length(ns_WOT), k=1:1];

    MSEs_fixed_dt_WOT = dropdims(mean((D_estimates_fixed_dt_WOT .- true_D).^2, dims = 1), dims = 1)
    std_errs_fixed_dt_WOT = dropdims(std((D_estimates_fixed_dt_WOT .- true_D).^2, dims = 1), dims = 1) ./ sqrt(num_trials)

    scatter!(ns_WOT, MSEs_fixed_dt_WOT[:, 1],
        yerr = std_errs_fixed_dt_WOT,
        label = L"$\textrm{MSE}(\mathcal{D}_{WOT})$",
        marker = marker_size,
    )
end


if save_pdfs
    savefig(figure_path*"MSE_vs_n.pdf")
end





# Part c): MSE with fixed n, varying dt

n = 2 ^ 9
dts = 0.02:0.04:0.5


@time D_estimates_fixed_n = [run_trial(true_D,
        n,
        dims,
        flow_type = :linear,
        linear_velocity_matrix = linear_velocity_matrix,
        neighbor_average = false,
        velocity_variance_scale = 0,
        estimators = estimators,
        dt_time_series = dt
        ) for i in 1:num_trials, dt in dts];




D_estimates_fixed_n = [D_estimates_fixed_n[i,j][k] for i=1:num_trials, j=1:length(dts), k=1:num_estimators];



MSEs_fixed_n = dropdims(mean((D_estimates_fixed_n .- true_D).^2, dims = 1), dims = 1)
std_errs_fixed_n = dropdims(std((D_estimates_fixed_n .- true_D).^2, dims = 1), dims = 1) ./ sqrt(num_trials)


scatter(dts, MSEs_fixed_n[:, 1],
    yerr = std_errs_fixed_n[:, 1],
    label = L"$\textrm{MSE(\widehat\mathcal{D})}$",
    yscale = :log10,
    xlabel = L"\delta t",
     ylabel = L"\textrm{MSE}",
 marker = marker_size,
 shape = :diamond,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 size = plot_size,
 title = L"$\textrm{c})$",
 titleloc = :left,
 titlefont = title_font_size,
)

if run_WOT
    scatter!(dts, MSEs_fixed_n[:, 2],
        yerr = std_errs_fixed_n[:, 2],
        label = L"$\textrm{MSE}(\mathcal{D}_{WOT})$",
        marker = marker_size,
    )
end

#todo: add error bars

if save_pdfs
    savefig(figure_path*"MSE_vs_dt.pdf")
end





# Part d): MSE with dt ~ n^(-1/4)


# Same n range as above

reference_n = 2

@time D_estimates_varying_both = [run_trial(true_D,
        n,
        dims,
        flow_type = :linear,
        linear_velocity_matrix = linear_velocity_matrix,
        neighbor_average = false,
        velocity_variance_scale = 0,
        dt_time_series = dt_time_series*(n / reference_n)^(-0.25), 
        estimators = [estimate_D],
        ) for i in 1:num_trials, n in ns];





D_estimates_varying_both = [D_estimates_varying_both[i,j][k] for i=1:num_trials, j=1:length(ns), k=1:1];


MSEs_varying_both = dropdims(mean((D_estimates_varying_both .- true_D).^2, dims = 1), dims = 1)
std_errs_varying_both = dropdims(std((D_estimates_varying_both .- true_D).^2, dims = 1), dims = 1) ./ sqrt(num_trials)

scatter(ns, MSEs_varying_both[:, 1],
    yerr = std_errs_varying_both,
    label = L"$\textrm{MSE(\widehat\mathcal{D})}$",
    xscale = :log10,
    yscale = :log10,
    xlabel = "\$n\$",
     ylabel = L"\textrm{MSE}",
 marker = marker_size,
 shape = :diamond,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 size = plot_size,
 title = L"$\textrm{d})$",
 titleloc = :left,
 titlefont = title_font_size,
)

if run_WOT
    @time D_estimates_varying_both_WOT = [run_trial(true_D,
            n,
            dims,
            flow_type = :linear,
            linear_velocity_matrix = linear_velocity_matrix,
            neighbor_average = false,
            velocity_variance_scale = 0,
            estimators = [estimate_D_wot],
            dt_time_series = dt_time_series*(n / reference_n)^(-0.25),
            ) for i in 1:num_trials, n in ns_WOT];
    D_estimates_varying_both_WOT = [D_estimates_varying_both_WOT[i,j][k] for i=1:num_trials, j=1:length(ns_WOT), k=1:1];

    MSEs_varying_both_WOT = dropdims(mean((D_estimates_varying_both_WOT .- true_D).^2, dims = 1), dims = 1)
    std_errs_varying_both_WOT = dropdims(std((D_estimates_varying_both_WOT .- true_D).^2, dims = 1), dims = 1) ./ sqrt(num_trials)

    
    scatter!(ns_WOT, MSEs_varying_both_WOT[:, 1],
        yerr = std_errs_varying_both_WOT,
        label = L"$\textrm{MSE}(\mathcal{D}_{WOT})$",
        marker = marker_size,
    )
end

plot!(ns,
    ns.^(-1/2)*MSEs_varying_both[1, 1]*ns[1]^(1/2),
    label = L"n^{-1/2}\,\,\textrm{ scaling}",
)



if save_pdfs
    savefig(figure_path*"MSE_vs_n_varying_dt.pdf")
end
