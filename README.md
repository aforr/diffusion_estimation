# Diffusion matrix estimation

This repository contains Python and Julia scripts to reproduce figures in an accompanying paper on estimating the diffusion term of a Fokker-Planck equation given samples from the marginals and an estimate of the drift term.

To recreate Figure 1, 3, or 4, create a Julia environment with the appropriate packages and run the corresponding script. PDF files corresponding to figure X will be saved in `Figures/Figure_X/`.

To recreate Figure 5, first create a Python environment with the appropriate packages and run `figure_5_prep.py` to download and process the hippocampus dataset. Then run `figure_5.jl` to produce the figure panels.

Julia packages required for Figures 3 and 4 are listed in `dependencies.jl`. Dependencies for Figures 1 and 5 are at the top of the corresponding scripts.