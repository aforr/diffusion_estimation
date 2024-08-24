using LaTeXStrings
using LinearAlgebra
using Plots
using QuadGK

figure_path = "Figures/Figure_1/"
if !isdir(figure_path)
	mkpath(figure_path)
end

tick_font_size = 14
label_font_size = 18
title_font_size = 18
line_width = 3
plot_size = (600, 450)

function create_potential(locations, widths, scales)
	# Returns a function computing a potential 
	# 
	# Parameters:
	# 	locations		Approximate locations of potential minima
	#	widths			Length scales of potential minima
	#	scales			Coefficients of Gaussian terms in potential

    return x -> x^2/400 + sum([s*exp(-((x-l)/w)^2)
            for (s, l, w) in zip(scales, locations, widths)])
end
    
    
    
potential = create_potential([-1, 5], [0.1, 5], [-1.5, -2])
xs = -5:0.01:10
plot(xs, potential.(xs),
 ylim = [-2.2, 0.2],
 legend = nothing,
 xlabel = "\$x\$",
 ylabel = "\$u(x)\$",
 linewidth = line_width,
 tickfont = tick_font_size,
 guidefont = label_font_size,
 yticks = [-2, -1, 0],
 xticks = [-4, 0, 4, 8],
 size = plot_size,
 title = L"$\textrm{a})$",
 titleloc = :left,
 titlefont = title_font_size,
 )
savefig(figure_path*"1d_two_well_potential.pdf")


plot()
for D in [0.1, 1, 10]
    normalization = quadgk(x -> exp(-potential(x)/D), -Inf, Inf)[1]
    plot!(xs, exp.(-potential.(xs)./D)/normalization, label = "\$D = \$"*string(D))
end
plot!()



function stationary_pos_prob(D)
	# Computes the probability x>0 in the stationary distribution
    pos_weight = quadgk(x -> exp(-potential(x)/D), 0, Inf)[1]
    neg_weight = quadgk(x -> exp(-potential(x)/D), -Inf, 0)[1]
    
    return pos_weight/(pos_weight+neg_weight)
end


Ds = 10 .^ (-2.5:0.05:2)
print(maximum(stationary_pos_prob.(Ds)))
plot(Ds, stationary_pos_prob.(Ds),
    xscale = :log10,
    legend = nothing,
    ylabel = L"P(x>0)",
    xlabel = L"\mathcal{D}",
    ylim = [0,1],
	linewidth = line_width,
	tickfont = tick_font_size,
	guidefont = label_font_size,
	size = plot_size,
	title = L"\textrm{b})",
	titleloc = :left,
	titlefont = title_font_size,
	)

savefig(figure_path*"stationary_prob_vs_D.pdf")