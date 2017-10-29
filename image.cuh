//raytracer.mustafaisik.net//

#pragma once

#include <glm/glm.hpp>

#include <string>

#include "cuda_runtime.h"

class Image
{
public:
    __host__ Image(int image_width, int image_height);
    __host__ ~Image();
    __host__ Image(const Image&) = delete;
    __host__ Image(Image&&) = delete;
    __host__ Image& operator=(const Image&) = delete;
    __host__ Image& operator=(Image&&) = delete;

    __host__ void save(const std::string& filepath) const;

    __device__ void setPixel(int pixel_index, const glm::vec3& color);

    //Getters
    __host__ unsigned char* Image::get_image() const
    {
        return m_image;
    }

    __device__ inline int get_width() const
    {
        return m_width;
    }

    __device__ inline int get_height() const
    {
        return m_height;
    }
    
private:
    unsigned char* m_image;
    int m_width, m_height;
};