import numpy as np
import ot
from pathlib import Path
import scanpy as sc
import scvelo as scv


# load data from scvelo


if Path('data/DentateGyrus.h5ad').exists():
	adata = sc.read('data/DentateGyrus.h5ad')
else:
	# load and preprocess data
	adata = scv.datasets.dentategyrus_lamanno()
	
	
	scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

	sc.pp.pca(adata)
	sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
	scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

	# This step may take around an hour
	scv.tl.recover_dynamics(adata)


	scv.tl.velocity(adata, mode='dynamical')
	scv.tl.velocity_graph(adata)


	def age_to_time(age):
		if age == "P0":
			return 0
		elif age == "P5":
			return 1
		else:
			raise ValueError("Unknown age: ", age)

	adata.obs['time'] = adata.obs['Age'].apply(age_to_time)

	scv.pl.velocity_embedding_stream(adata, basis='tsne', color = 'time')

	adata.write('data/DentateGyrus.h5ad')



np.savetxt("data/DentateGyrus_tsne.csv", adata.obsm["X_tsne"], delimiter = ",")
np.savetxt("data/DentateGyrus_velocity_tsne.csv", adata.obsm["velocity_tsne"], delimiter = ",")
np.savetxt("data/DentateGyrus_time.csv", adata.obs["time"], delimiter = ",", fmt="%d")






















times = np.sort(adata.obs["time"].unique())
state_means = {}
velocity_means = {}
pop_covariances = {}

for time in times:
    state_means[time] = np.mean(adata[adata.obs["time"] == time, adata.var["velocity_genes"]].X, axis = 0)
    velocity_means[time] = np.mean(adata[adata.obs["time"] == time, adata.var["velocity_genes"]].layers["velocity"],
                                   axis = 0)
    
    
    pop_covariances[time] = np.cov(adata[adata.obs["time"] == time, adata.var["velocity_genes"]].X.todense(), rowvar = False)



num_dimensions = pop_covariances[0].shape[0]
velocity_timescales = {}
pushforward_samples = {}
pushforward_covariances = {}
timescales = range(12)

for timescale in timescales:
    pushforward_samples[timescale] = {}
    pushforward_covariances[timescale] = {}

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i+1]
        state_difference = state_means[t1] - state_means[t0]
        
        inner_product = np.dot(state_difference, velocity_means[t0])
        velocity_timescales[t0] = inner_product / np.linalg.norm(velocity_means[t0])**2
        velocity_timescales[t0] = velocity_timescales[t0][0,0]
        
        
        pushforward_samples[timescale][t0] = (adata[adata.obs["time"] == t0, adata.var["velocity_genes"]].X
                                   + timescale*velocity_timescales[t0]*adata[adata.obs["time"] == t0, adata.var["velocity_genes"]].layers["velocity"]
        )
        
        pushforward_covariances[timescale][t0] = np.cov(pushforward_samples[timescale][t0], rowvar = False)




# calculate covariance D*dt

D_dt_estimates = np.array(
	[np.trace(pop_covariances[1]-pushforward_covariances[timescale][0]) / num_dimensions / 2
	for timescale in timescales]
)

# calculate covariance bound on D*dt

D_dt_bound = np.trace(pop_covariances[1]) / num_dimensions / 2

# calculate WOT D*dt without PCA

pairwise_distances_no_PCA = ot.utils.dist(adata[adata.obs["time"] == 0, adata.var["velocity_genes"]].X.todense(),
                                    adata[adata.obs["time"] == 1, adata.var["velocity_genes"]].X.todense()
                                  )


WOT_D_dt_no_PCA = np.median(np.array(pairwise_distances_no_PCA))*0.05

# calculate WOT D*dt with PCA

pairwise_distances_with_PCA = ot.utils.dist(adata[adata.obs["time"] == 0, adata.var["velocity_genes"]].obsm["X_pca"],
                                    adata[adata.obs["time"] == 1, adata.var["velocity_genes"]].obsm["X_pca"]
                                  )


WOT_D_dt_with_PCA = np.median(np.array(pairwise_distances_with_PCA))*0.05



# Write to csv for import into Julia
np.savetxt("data/timescales.csv", timescales, delimiter = ",")
np.savetxt("data/D_dt_estimates.csv", D_dt_estimates, delimiter = ",")
np.savetxt("data/D_dt_bound_and_WOT.csv", [D_dt_bound, WOT_D_dt_no_PCA, WOT_D_dt_with_PCA], delimiter = ",")