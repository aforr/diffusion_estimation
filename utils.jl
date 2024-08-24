# Utility functions for data processing
include("dependencies.jl")



##########################
# Nearest neighbor averaging of velocities
# (not used in original figures)
##########################

function convert_to_static(x)
    l = length(x)
    return SVector{l}(x)
end

function get_neighbors(xs, k)
    static_xs = [convert_to_static(x) for x in eachcol(xs)]
    neighbor_tree = KDTree(static_xs; leafsize = 10)
    neighbor_indices, neighbor_distances = knn(neighbor_tree, static_xs, k);
    
    return neighbor_indices, neighbor_distances
end

function get_symmetric_neighbors(xs, k)
    d, n_samples = size(xs)
    symmetric_neighbors, tmp = get_neighbors(xs, k)
    for i in 1:n_samples
        for j in symmetric_neighbors[i]
            if !(i in symmetric_neighbors[j])
                symmetric_neighbors[j] = vcat(symmetric_neighbors[j], [i])
            end
        end
    end
    for i in 1:n_samples
        sort!(symmetric_neighbors[i])
    end
    
    return symmetric_neighbors
end


function get_neighbor_differences(xs, neighbors)
    n_samples = length(xs)
    neighbor_differences = [[xs[:,j] - xs[:,i] for j in neighbors[i]] for i in 1:n_samples];
    return neighbor_differences
end



function compute_neighbor_weights(neighbors)
    n_samples = length(neighbors)
    weights = [ones(length(neighbors[i]))/length(neighbors[i]) for i in 1:n_samples]
    return weights
end



function estimate_v(dxs, x_neighbors, x_weights)
    neighbor_dxs = hcat([dxs[:,i] for i in x_neighbors]...)
    return neighbor_dxs*x_weights
end

function estimate_vs(dxs, neighbors)
    weights = compute_neighbor_weights(neighbors)
    return [estimate_v(dxs, neighbors[i], weights[i]) for i in 1:length(neighbors)]
end






##########################
# Solving linear systems for diffusion matrices
##########################

struct MatrixSystemArray{T} <: AbstractArray{T, 2}
    # The linear operator applied to X (where X is considered as a vector)
    # in the system of equations
    # base_matrix*X + X*base_matrix' = B
    #
    # This can be used to compute x = A\b
    # where x = X[:] and b = B[:]
    base_matrix::Array{T, 2}
end

Base.size(A::MatrixSystemArray) = length(A.base_matrix) .* (1, 1)


function Base.getindex(A::MatrixSystemArray, i::Int, j::Int)
    A_row_indices = linear_index_to_row_col(i, Base.size(A.base_matrix)[1])
    A_col_indices = linear_index_to_row_col(j, Base.size(A.base_matrix)[1])
    
    s = 0
    if A_row_indices[1] == A_col_indices[1]
        s = s + A.base_matrix[A_row_indices[2], A_col_indices[2]]
    end
    if A_row_indices[2] == A_col_indices[2]
        s = s + A.base_matrix[A_row_indices[1], A_col_indices[1]]
    end
    return s
end

function linear_index_to_row_col(i, n_rows)
    row_index = ((i-1) % n_rows) + 1
    col_index = Int64(floor((i-1) / n_rows) + 1)
    return row_index, col_index
end




function solve_transpose_system(A, B)
    # Solves the linear system A*X + X*A^T = B
    # and returns the matrix X
    # B is assumed symmetric, so X will be symmetric
    # but A is not necessarily symmetric
    
    M = MatrixSystemArray(A)
    v = M \ B[:]
    return reshape(v, size(B))
end



