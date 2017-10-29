//raytracer.mustafaisik.net//

#pragma once

#include "ray.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

class Instance
{
public:
    __host__ Instance::Instance()
        : m_inverse_transformation(1.0f)
        , m_transformation(1.0f)
        , m_det_sign(1.0f)
    {}

    __host__ Instance::Instance(const glm::mat4& transformation)
        : m_inverse_transformation(glm::inverse(transformation))
        , m_transformation(transformation)
        , m_det_sign(glm::determinant(transformation))
    {
        m_det_sign = m_det_sign > 0.0f ? 1.0f : -1.0f;
    }

    __device__ inline Ray rayToObjectSpace(const Ray& ray) const
    {
        return Ray(m_inverse_transformation * glm::vec4(ray.get_origin(), 1.0f),
            m_inverse_transformation * glm::vec4(ray.get_direction(), 0.0f));
    }

    __device__ inline glm::vec3 normalToWorldSpace(const glm::vec3& normal) const
    {
        return glm::normalize(glm::vec3(glm::transpose(m_inverse_transformation) * glm::vec4(normal, 0.0f)));
    }

    __device__ inline void tangentToWorldSpace(glm::vec3& dpdu, glm::vec3& dpdv) const
    {
        dpdu = (m_transformation * glm::vec4(dpdu, 0.0f)) * m_det_sign;
        dpdv = (m_transformation * glm::vec4(dpdv, 0.0f));
    }

private:
    glm::mat4 m_inverse_transformation;
    glm::mat4 m_transformation;
    float m_det_sign;
};