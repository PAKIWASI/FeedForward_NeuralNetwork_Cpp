#pragma once

#include "layer.hpp"
#include <vector>


class Hidden_Layer : public Layer
{
private:
    void activation(const std::vector<float>& z, std::vector<float>& a) override;  
    void ReLu_deriv(const std::vector<float>& z, std::vector<float>& da_dz);


protected:
    
public:
    Hidden_Layer(int input_size, int output_size);
    void init_weights_biases() override; // HE INIT FOR HIDDEN LAYERS
    void calc_deriv(std::vector<float>& upstream_grad) override;

};
