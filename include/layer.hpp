#pragma once

#include <random>
#include <vector>


class Layer 
{
protected:
    std::vector<float> input;                // x     (1 x m) comes from prev later
    std::vector<std::vector<float>> weights; // W     (m x n)
    std::vector<float> biases;               // b     (1 x n)
    std::vector<float> pre_act_output;       // z     (1 x m)x(m x n)->(1 x n) z = xW + b
    std::vector<float> act_output;           // a     (1 x n) a = f(z)
     

    // GRADIENTS
    // first we get dL/da as upstream gradient
    // we use this to calculate dL/dz, which is req to calc W,b and downstream gradients
    // Local Gradient
    //std::vector<float> da_dz; // da/dz x dL/dz (element wise) = dL/dz (1 x n) (it's actually a jacobian)
    // Gradients to update Weights and biases
    std::vector<std::vector<float>> dL_dW; //  = dz/dW x dL/dz = (x)T x dL/dz  (m x 1) x (1 x n) -> (m x n) 
    std::vector<float> dL_db;  // = dL/dz x dz/db = dL/dz x 1  ->(1 x n)
    // Downstream Gradient
    std::vector<float> dL_dx;  // (1 x m)  goes to the prev layer

    const int input_size;   // m
    const int output_size;  // n
 

    virtual void activation(const std::vector<float>& z, std::vector<float>& a) = 0;  

    std::mt19937 rng;
    
public:
    Layer(int input_size, int output_size);
    virtual ~Layer() = default;  // Ensures proper vtable generation
    
    virtual void init_weights_biases() = 0;
    void calc_output(const std::vector<float>& input);
    virtual void calc_deriv(std::vector<float>& upstream_grad) = 0;
    void update_weights_biases( float LEARING_RATE);

    virtual void cleanup();

    std::vector<float>& get_output() { return act_output; }
    std::vector<float>& get_upstream() { return dL_dx; }//goes to prev layer as upstream grad
};
