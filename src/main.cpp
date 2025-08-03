#include "neural_network.hpp"

int main()
{
    const float LEARNING_RATE = 0.01F;
    const std::vector<int> shape = {32, 16 };

    Neural_Network nn(shape, LEARNING_RATE);
    
    nn.train();
    nn.test();
    return 69;
}

/* Notes:
 * need to clean up w & b gradients after every iteration
 * just see which vars are generally needed fresh each iteration
 * check that pointer logic for dL/db
 * check the activation functins, derivs and the clac derivs
 */
