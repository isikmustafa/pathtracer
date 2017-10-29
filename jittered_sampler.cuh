#pragma once

#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "curand_kernel.h"

class JitteredSampler
{
public:
    __host__ JitteredSampler(int dim)
        : m_start_dim(dim)
        , m_dim(dim)
        , m_offset(0)
        , m_res(1.0f / dim)
    {}

    __device__ inline glm::vec2 getSample(curandState& rand_state, int stratum)
    {
        if (stratum - m_offset == m_dim * m_dim)
        {
            ++m_dim;
            m_offset = stratum;
            m_res = 1.0f / m_dim;
        }
        else if (stratum == 0)
        {
            m_dim = m_start_dim;
            m_offset = 0;
            m_res = 1.0f / m_dim;
        }

        stratum -= m_offset;
        auto i = (stratum % m_dim) * m_res;
        auto j = (stratum / m_dim) * m_res;

        return glm::vec2(i + curand_uniform(&rand_state) * m_res, j + curand_uniform(&rand_state) * m_res);
    }

private:
    int m_start_dim;
    int m_dim;
    int m_offset;
    float m_res;
};