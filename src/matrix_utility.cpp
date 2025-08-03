#include "matrix_utility.hpp"
#include <stdexcept>

//   (1 x m) x (m x n) -> (1 x n)
void vector_xply_matrix(const std::vector<float>& vec,
                        const std::vector<std::vector<float>>& matrix,
                        std::vector<float>& output)
{
    int m = static_cast<int>(vec.size());
    int n = static_cast<int>(matrix[0].size());
    int p = static_cast<int>(matrix.size());
    if (m != p)
    {
        throw std::runtime_error("vec matrix xply not appliable(dimentions dont add up)\n");
        return;
    }

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            output[i] += vec[j] * matrix[j][i];
        }
    }
}


// dL/dW = dz/dW x dL/dz = xT x dL/dz (m x 1) x (1 x n) -> (m x n)
void vector_transpose_xply_vector( const std::vector<float>& vec1,
                                   const std::vector<float>& vec2,
                                   std::vector<std::vector<float>>& matrix )
{
    int m = static_cast<int>(vec1.size());  // to be transposed  
    int n = static_cast<int>(vec2.size());

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = vec1[i] * vec2[j]; 
        }
    }
}

// dL/dx = dL/dz x dz/dx = dL/dz x WT (1 x n) x (n x m) ->(1 x m)
void vector_xply_matrix_transpose( const std::vector<float>& vec,
                                   const std::vector<std::vector<float>>& matrix,
                                   std::vector<float>& output     )
{
    int n = static_cast<int>(vec.size());
    int m = static_cast<int>(matrix.size());
    int n2 = static_cast<int>(matrix[0].size());

    if(n != n2) 
    {
        throw std::runtime_error("dimentions dont add up\n");
        return;
    }
    
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            output[j] += vec[i] * matrix[j][i];
        }
    }
}
