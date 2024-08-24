using CSV
using DataFrames
using LaTeXStrings
using Plots
gr()

tick_font_size = 14
legend_font_size = 14
label_font_size = 18
title_font_size = 18
line_width = 3
marker_size = 7

plot_size = (600, 450)

figure_path = "Figures/Figure_4/"
if !isdir(figure_path)
	mkpath(figure_path)
end

# Load data from preprocessing script (dentate_gyrus_data_prep.py)
tsne_coords = CSV.read("data/DentateGyrus_tsne.csv",
 	DataFrame,
 	header = 0
)

tsne_v = CSV.read("data/DentateGyrus_velocity_tsne.csv",
	DataFrame,
	header = 0
)

cell_time = CSV.read("data/DentateGyrus_time.csv",
	DataFrame,
	header = 0,
	types = Bool,
)


# Scatter plot in tsne coordinates
scatter(tsne_coords[.!cell_time[!, 1], 1], tsne_coords[.!cell_time[!, 1], 2],
 alpha = 0.1,
 label = "P0",
 xlabel = "t-SNE 1",
 ylabel = "t-SNE 2",
 marker = marker_size,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 size = plot_size,
 title = L"$\textrm{a})$",
 titleloc = :left,
 titlefont = title_font_size,
 )

scatter!(tsne_coords[cell_time[!, 1], 1], tsne_coords[cell_time[!, 1], 2],
 alpha = 0.1, 
 marker = marker_size,
 label = "P5")



savefig(figure_path*"DentateGyrus_tsne.pdf")

# Diffusion coefficient plot





timescales = CSV.read("data/timescales.csv",
	DataFrame,
	header = 0
)[!,1]

D_dt_estimates = CSV.read("data/D_dt_estimates.csv",
DataFrame,
header = 0
)[!,1]

D_dt_bound_and_WOT = CSV.read("data/D_dt_bound_and_WOT.csv",
	DataFrame,
	header = 0
)

D_dt_bound = D_dt_bound_and_WOT[1,1]
D_dt_WOT_no_PCA = D_dt_bound_and_WOT[2,1]
D_dt_WOT_with_PCA = D_dt_bound_and_WOT[3,1]


plot(timescales[1:end-2], D_dt_estimates[1:end-2],
 label = L"$\widehat{\mathcal{D}}\delta t$",
 xlabel = L"$\alpha/\hat\alpha$",
 ylabel = L"$\mathcal{D}$",
 yscale = :log10,
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 legendfont = legend_font_size,
 size = plot_size,
 title = L"$\textrm{b})$",
 titleloc = :left,
 titlefont = title_font_size,
 )

plot!([minimum(timescales), maximum(timescales)],
	D_dt_bound*ones(2),
	linestyle = :dot,
	linewidth = line_width,
	label = L"$\mathcal{D}\delta t \, \textrm{bound}$",
)

plot!([minimum(timescales), maximum(timescales)],
	D_dt_WOT_with_PCA*ones(2),
	linestyle = :dash,
	linewidth = line_width,
	label = L"$\mathcal{D}_{WOT}\delta t$",
)

savefig(figure_path*"DentateGyrus_D_estimates.pdf")