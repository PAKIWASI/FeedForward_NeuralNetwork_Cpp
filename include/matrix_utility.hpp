#pragma once
#include <vector>

void vector_xply_matrix(const std::vector<float>& vec,
                        const std::vector<std::vector<float>>& matrix,
                        std::vector<float>& output);


// dL/dW = dz/dW x dL/dz = xT x dL/dz (m x 1) x (1 x n)
void vector_transpose_xply_vector( const std::vector<float>& vec1,
                                   const std::vector<float>& vec2,
                                   std::vector<std::vector<float>>& matrix);

void vector_xply_matrix_transpose(
                               const std::vector<float>& vec,
                               const std::vector<std::vector<float>>& matrix,
                               std::vector<float>& output     );
