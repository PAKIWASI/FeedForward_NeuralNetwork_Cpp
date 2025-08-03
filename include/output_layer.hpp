#pragma once

#include "layer.hpp"
#include <vector>

class Outer_Layer : public Layer
{
private:
    std::vector<float> dL_dz;

    void activation(const std::vector<float>& z, std::vector<float>& a) override;  
    void softmax_cross_entropy_deriv(
        const std::vector<float>& predicted,
        const std::vector<float>& true_label,
        std::vector<float>& dL_dz   );
    
public:
    Outer_Layer(int input_size, int output_size);

    void init_weights_biases() override; // Xavior init for outer layer 
    void calc_deriv(std::vector<float>& true_label) override;
 
    int get_prediction();
};
