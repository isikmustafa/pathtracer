//raytracer.mustafaisik.net//

#pragma once

#include <glm/glm.hpp>

#include "cuda_runtime.h"

class Ray
{
public:
    __device__ Ray(const glm::vec3& origin, const glm::vec3& direction)
        : m_origin(origin)
        , m_direction(direction)
    {}

    __device__ inline glm::vec3 getPoint(float distance) const
    {
        return m_origin + distance * m_direction;
    }

    //Getters
    __device__ inline const glm::vec3& get_origin() const
    {
        return m_origin;
    }

    __device__ inline const glm::vec3& get_direction() const
    {
        return m_direction;
    }

private:
    glm::vec3 m_origin;
    glm::vec3 m_direction;
};