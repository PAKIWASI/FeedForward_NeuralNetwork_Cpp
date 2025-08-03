#include "hidden_layer.hpp" 
#include "matrix_utility.hpp"
#include <cmath>  // Add for std::sqrt

Hidden_Layer::Hidden_Layer(int input_size, int output_size) 
        : Layer(input_size, output_size)
{
    dL_dz.resize(output_size);
    da_dz.resize(output_size, 0);
}

void Hidden_Layer::init_weights_biases() // HE INIT FOR HIDDEN LAYERS
{
    float stddev = std::sqrt(2.0F / static_cast<float>(input_size));
    std::normal_distribution<float> dist(0.0F, stddev);

    
    for (int i = 0; i < input_size; ++i) 
    {
        for (int j = 0; j < output_size; ++j) 
        {
            weights[i][j] = dist(rng);
        }
    }
// biases set to zero in constructor
}

void Hidden_Layer::activation(const std::vector<float>& z, std::vector<float>& a) 
{
    for (int i = 0; i < output_size; i++)
    {
        if (z[i] >= 0) { a[i] = z[i]; }
        else           { a[i] = 0; }
    }
}

void Hidden_Layer::ReLu_deriv(const std::vector<float>& z, std::vector<float>& da_dz) 
{
    for (int i = 0; i < output_size; i++)
    {
        if (z[i] >= 0) { da_dz[i] = 1; }
        else           { da_dz[i] = 0; }
    }
}

void Hidden_Layer::calc_deriv(std::vector<float>& upstream_grad)
{
        //activation deriv
    ReLu_deriv(pre_act_output, da_dz);
    
    for (int i = 0; i < output_size; i++)
    {
        dL_dz[i] = upstream_grad[i] * da_dz[i];  // element wise (jacobian)
    }

    // dL/dW = dz/dW x dL/dz = xT x dL/dz (m x 1) x (1 x n)
    vector_transpose_xply_vector(input, dL_dz, dL_dW);

    // dL/db = dL/dz x dz/db = dL/dz
    dL_db = dL_dz;

    // dL/dx = dL/dz x dz/dx = dL/dz x WT (1 x n) x (n x m) -> (1 x m)
    vector_xply_matrix_transpose(dL_dz, weights, dL_dx);
}
