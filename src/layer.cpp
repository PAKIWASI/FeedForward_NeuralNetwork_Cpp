#include "layer.hpp"
#include "matrix_utility.hpp"


Layer::Layer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size),rng(std::random_device{}())  // Initialize with random seed
{
    weights.resize(input_size, {});
    for(auto& v : weights) { v.resize(output_size, 0); }
    biases.resize(output_size, 0);
    pre_act_output.resize(output_size, 0);
    act_output.resize(output_size, 0);

    dL_dx.resize(input_size, 0);   // a ref to this will go to prev layer
    dL_dW.resize(input_size, {});
    for(auto& v : dL_dW) { v.resize(output_size, 0); }
}



void Layer::calc_output(const std::vector<float>& input)
{
    this->input = input;
    vector_xply_matrix(input, weights, pre_act_output); 

    for (int i = 0; i < output_size; i++)
    {
        pre_act_output[i] += biases[i];
    }

    activation(pre_act_output, act_output);
}


void Layer::update_weights_biases( float LEARING_RATE)
{
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            weights[i][j] += -1 * (LEARING_RATE * dL_dW[i][j]);
        }
    }

    for (int i = 0; i < output_size; i++)
    {
        biases[i] += -1 * (LEARING_RATE * (dL_db[i]));
    }
}

void Layer::cleanup()
{
    std::fill(pre_act_output.begin(), pre_act_output.end(), 0);
    std::fill(act_output.begin(), act_output.end(), 0);
    for (auto& row : dL_dW) {
        std::fill(row.begin(), row.end(), 0);
    }
    std::fill(dL_db.begin(), dL_db.end(), 0);
    std::fill(dL_dx.begin(), dL_dx.end(), 0);
}
