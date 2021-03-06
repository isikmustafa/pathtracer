//raytracer.mustafaisik.net//

#pragma once

#include <iostream>

#include "cuda_runtime.h"

static void handleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        system("PAUSE");
    }
}

#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))