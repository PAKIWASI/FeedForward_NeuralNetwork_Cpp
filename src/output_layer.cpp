#include "output_layer.hpp"
#include "matrix_utility.hpp"

#include <cmath>

Outer_Layer::Outer_Layer(int input_size, int output_size)
    : Layer(input_size, output_size)
{
    dL_dz.resize(output_size);
}

void Outer_Layer::init_weights_biases() // Xavior init
{
    float limit = std::sqrt(6.0F / (static_cast<float>(input_size + output_size)));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < input_size; ++i) 
    {
        for (int j = 0; j < output_size; ++j) 
        {
            weights[i][j] = dist(rng);
        }
    }
// bias set to zero in constructor
}

// SOFTMAX
void Outer_Layer::activation(const std::vector<float>& z, std::vector<float>& a) 
{

    float max = z[0];
    for (int i = 0; i < output_size; i++)
    {
        if (z[i] > max)
        {
            max = z[i];
        }
    }
    float sum = 0;

    for (int i = 0; i < output_size; i++)
    {
        a[i] = std::exp(z[i] - max);
        sum += a[i];  // fixed !
    }
    for (int i = 0; i < output_size; i++)
    {
        a[i] /= sum;
    }
}

// SOFTMAX + CROSS ENTROPY
void Outer_Layer::softmax_cross_entropy_deriv(const std::vector<float>& predicted,
                                            const std::vector<float>& true_label,
                                            std::vector<float>& dL_dz) 
{
    // p = predicted vector , y = true label vector (hot coded)
    for (int i = 0; i < output_size; i++)
    {
        dL_dz[i] = predicted[i] - true_label[i];
    }
}

void Outer_Layer::calc_deriv(std::vector<float>& true_label) // for outer layer , this is just the true label
{

    softmax_cross_entropy_deriv(act_output,true_label,dL_dz);

    // dL/dW = dz/dW x dL/dz = xT x dL/dz (m x 1) x (1 x n)
    vector_transpose_xply_vector(input, dL_dz, dL_dW);

    // dL/db = dL/dz x dz/db = dL/dz
    dL_db = dL_dz;

    // dL/dx = dL/dz x dz/dx = dL/dz x WT (1 x n) x (n x m) -> (1 x m)
    vector_xply_matrix_transpose(dL_dz, weights, dL_dx);
}

int Outer_Layer::get_prediction()
{
    float max = -9999;
    int index = -1;
    for(int i = 0; i < output_size; i++) 
    {
        if (act_output[i] > max)
        {
            max = act_output[i];
            index = i;
        }
    }
    return index;
}
