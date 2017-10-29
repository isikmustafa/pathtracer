//raytracer.mustafaisik.net//

#pragma once

#include <glm/glm.hpp>

#include "cuda_runtime.h"

extern __constant__ float gPerlinGradients[36];
extern __constant__ int gPerlinPermutation[512];

class Texture
{
public:
    enum TextureType
    {
        IMAGE,
        PERLIN
    };
    enum FilterMode
    {
        NEAREST = cudaFilterModePoint,
        BILINEAR = cudaFilterModeLinear
    };
    enum AddressMode
    {
        REPEAT = cudaAddressModeWrap,
        CLAMP = cudaAddressModeClamp
    };
    enum PerlinMode
    {
        PATCH,
        VEIN
    };
    struct SampleParams
    {
        //Sample parameters
        union
        {
            struct
            {
                FilterMode filter_mode;
                AddressMode address_mode;
            };
            struct
            {
                PerlinMode perlin_mode;
                float scaling_factor;
            };
        };

        SampleParams(FilterMode p_filter_mode = NEAREST, AddressMode p_address_mode = REPEAT)
            : filter_mode(p_filter_mode)
            , address_mode(p_address_mode)
        {}

        SampleParams(PerlinMode p_perlin_mode = PATCH, float p_scaling_factor = 1.0f)
            : perlin_mode(p_perlin_mode)
            , scaling_factor(p_scaling_factor)
        {}
    };

public:
    __host__ Texture(const SampleParams& sample_params, TextureType texture_type, cudaTextureObject_t texture)
        : m_sample_params(sample_params)
        , m_texture_type(texture_type)
        , m_texture(texture)
    {}

    __device__ inline glm::vec3 sampleImage(const glm::vec2& tex_coords) const
    {
        //Sample image texture
        float4 color = tex2D<float4>(m_texture, tex_coords.x, tex_coords.y);
        return glm::vec3(color.z, color.y, color.x); //BGR TO RGB
    }

    __device__ inline float samplePerlin(glm::vec3 point) const
    {
        //Sample perlin texture
        point *= m_sample_params.scaling_factor;
        auto xi = static_cast<int>(floorf(point.x)) & 255;
        auto yi = static_cast<int>(floorf(point.y)) & 255;
        auto zi = static_cast<int>(floorf(point.z)) & 255;

        auto xf = point.x - floorf(point.x);
        auto yf = point.y - floorf(point.y);
        auto zf = point.z - floorf(point.z);

        auto fade = [](float t)
        {
            return t * t * t * (t * (t * 6 - 15) + 10);
        };
        auto u = fade(xf);
        auto v = fade(yf);
        auto w = fade(zf);

        //Numbers to select gradient vectors at the edges.
        int v000, v010, v001, v011, v100, v110, v101, v111;
        v000 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi] + yi] + zi];
        v010 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi] + yi + 1] + zi];
        v001 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi] + yi] + zi + 1];
        v011 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi] + yi + 1] + zi + 1];
        v100 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi + 1] + yi] + zi];
        v110 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi + 1] + yi + 1] + zi];
        v101 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi + 1] + yi] + zi + 1];
        v111 = gPerlinPermutation[gPerlinPermutation[gPerlinPermutation[xi + 1] + yi + 1] + zi + 1];

        auto lerp = [](float a, float b, float t)
        {
            return a + t * (b - a);
        };
        auto grad = [](int hash, float x, float y, float z)
        {
            hash = (hash & 15) * 3;
            return glm::dot(glm::vec3(x, y, z), glm::vec3(gPerlinGradients[hash], gPerlinGradients[hash + 1], gPerlinGradients[hash + 2]));
        };

        float x1, x2, y1, y2;
        x1 = lerp(grad(v000, xf, yf, zf), grad(v100, xf - 1.0f, yf, zf), u);
        x2 = lerp(grad(v010, xf, yf - 1.0f, zf), grad(v110, xf - 1.0f, yf - 1.0f, zf), u);
        y1 = lerp(x1, x2, v);

        x1 = lerp(grad(v001, xf, yf, zf - 1.0f), grad(v101, xf - 1.0f, yf, zf - 1.0f), u);
        x2 = lerp(grad(v011, xf, yf - 1.0f, zf - 1.0f), grad(v111, xf - 1.0f, yf - 1.0f, zf - 1.0f), u);
        y2 = lerp(x1, x2, v);

        auto color = lerp(y1, y2, w);
        color = m_sample_params.perlin_mode == PATCH ? (color + 1.0f) / 2.0f : fabsf(color);

        return color;
    }

    //Getters
    __host__ __device__ inline TextureType get_texture_type() const
    {
        return m_texture_type;
    }

    __host__ inline cudaTextureObject_t get_texture() const
    {
        return m_texture;
    }

private:
    SampleParams m_sample_params;
    TextureType m_texture_type;
    cudaTextureObject_t m_texture;
};