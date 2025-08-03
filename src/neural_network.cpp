#include "neural_network.hpp"
#include "hidden_layer.hpp"
#include "output_layer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>



Neural_Network::Neural_Network(const std::vector<int>& hidden_layer_sizes, float LEARNING_RATE)
:LEARNING_RATE(LEARNING_RATE)
{      
    input.resize(input_size);

    // Create first hidden layer (input_size → hidden_layer_sizes[0])
    if (!hidden_layer_sizes.empty()) {
        layers.push_back(new Hidden_Layer(input_size, hidden_layer_sizes[0]));
    } else {
        // If no hidden layers, directly connect input → output
        layers.push_back(new Outer_Layer(input_size, output_size));
        return;
    }

    // Add remaining hidden layers
    for (size_t i = 0; i < hidden_layer_sizes.size() - 1; ++i) {
        layers.push_back(new Hidden_Layer(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]));
    }

    // Add output layer (last hidden → output_size)
    layers.push_back(new Outer_Layer(hidden_layer_sizes.back(), output_size));

    std::cout << "NN: " << layers.size() << " layers initialized\n";

    // Initialize weights
    for (auto& layer : layers) 
    {
        layer->init_weights_biases();
    }
    std::cout << "Weights and biases initialized\n";
}


void Neural_Network::train()
{
    int train_no = 0;
    std::ifstream file("../data/mnist_train.csv/mnist_train.csv");
    std::string line;

    std::cout << "TRAINING STARTED\n";
    if (!file.is_open())
    {
        std::cerr << "file not found\n";
        return;
    }
    
    while (std::getline(file, line))
    {
        // Reset gradients BEFORE forward/backward
        for (auto& layer : layers) 
            { layer->cleanup(); }

        std::stringstream ss(line);
        std::string cell;
        int i = 0;

        while (getline(ss, cell, ','))
        {
            if (i == 0) { tag = std::stoi(cell); i++; continue; }
            else { 
                input[i - 1] = std::stof(cell) / 255.0F; // normalise
            }
            i++;
        }

        if (train_no % 1000 == 0)
        { std::cout << "FORWARD PASS TRAIN NO " << train_no << '\n'; }
        train_no++;
                // FORWARD PASS      
        forward_pass();
                // BACKWARD PASS
        std::vector<float> true_label(output_size, 0);
        true_label[tag] = 1;
        backward_pass(true_label);
                // UPDATE PARAMETERS
        for (auto& i : layers)
        {
            i->update_weights_biases(LEARNING_RATE);
        }

    }
    file.close();

    std::cout << "TRAINING COMPLETED WITH " << train_no << " SAMPLES\n";
}

Neural_Network::~Neural_Network() 
{
    for (Layer*& layer : layers)  
    {   delete layer; }
    layers.clear();
}

void Neural_Network::test()
{
    int test_no = 0;
    int correct = 0;
    std::ifstream file("../data/mnist_test.csv/mnist_test.csv");
    std::string line;

    std::cout << "TESTING STARTING\n";
    if (!file.is_open())
    {
        std::cerr << "file not found\n";
        return;
    }
    
    while (std::getline(file, line))
    {
        for (auto& layer : layers) 
            { layer->cleanup(); }

        std::stringstream ss(line);
        std::string cell;
        int i = 0;

        while (getline(ss, cell, ','))
        {
            if (i == 0) { tag = std::stoi(cell); i++; continue; }
            input[i - 1] = std::stof(cell) / 255.0F;
            i++;
        }

        if (test_no % 1000 == 0)
        { std::cout << "FORWARD PASS TEST NO " << test_no << '\n'; }
        test_no++;
                // FORWARD PASS      
        forward_pass();
                // TEST AGAINST actual tags
        Layer* l = layers.back();
        Outer_Layer* ol = static_cast<Outer_Layer*>(l); //unsafe
        int predicted = ol->get_prediction();
        if (predicted == tag) { correct++; }
    }

    file.close();

    std::cout << "TESTING COMPLETED WITH " << test_no << " TESTS\n"; 
    std::cout << "CORRECT PREDICTIONS: " << correct << '\n';
    float accuracy = (static_cast<float>(correct) / static_cast<float>(test_no)) * 100.0F;
    std::cout << "ACCURACY: " << accuracy << "%\n";
}

void Neural_Network::forward_pass()
{
    layers[0]->calc_output(input); 
    for (unsigned int i = 1; i < layers.size(); i++)
    {
        layers[i]->calc_output(layers[i - 1]->get_output());
    }
}

void Neural_Network::backward_pass(std::vector<float>& true_label)
{
    layers.back()->calc_deriv(true_label); // outpur layer
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--)
    {
        layers[i]->calc_deriv(layers[i + 1]->get_upstream());
    }
}

