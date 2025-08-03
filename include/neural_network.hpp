#pragma once

#include "layer.hpp"

class Neural_Network
{
private:
    const float LEARNING_RATE;
    
    std::vector<Layer*> layers;

    const int input_size = 784; // MNIST image data size (tag + image)
    const int output_size = 10; // digits 0-9
    std::vector<float> input;
    int tag;
    
    void forward_pass();
    void backward_pass(std::vector<float>& true_label);
public:
    Neural_Network(const std::vector<int>& hidden_layer_sizes, float LEARNING_RATE);
    ~Neural_Network(); 

    void train();
    void test();
};
