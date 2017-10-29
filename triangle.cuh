//raytracer.mustafaisik.net//

#pragma once

#include "bbox.cuh"
#include "intersection.cuh"
#include "ray.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

class Triangle
{
public:
    __host__ Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2,
        const glm::vec2& texcoord0 = glm::vec2(), const glm::vec2& texcoord1 = glm::vec2(), const glm::vec2& texcoord2 = glm::vec2());
    __host__ Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2, const glm::vec3& normal0, const glm::vec3& normal1, const glm::vec3& normal2,
        const glm::vec2& texcoord0 = glm::vec2(), const glm::vec2& texcoord1 = glm::vec2(), const glm::vec2& texcoord2 = glm::vec2());
    __host__ BBox getBBox() const;

    __device__ inline bool intersect(const Ray& ray, Intersection& intersection) const
    {
        //Möller-Trumbore algorithm
        auto pvec = glm::cross(ray.get_direction(), m_edge2);
        auto inv_det = 1.0f / glm::dot(m_edge1, pvec);

        auto tvec = ray.get_origin() - m_v0;
        auto w1 = glm::dot(tvec, pvec) * inv_det;

        if (w1 < 0.0f || w1 > 1.0f)
        {
            return false;
        }

        auto qvec = glm::cross(tvec, m_edge1);
        auto w2 = glm::dot(ray.get_direction(), qvec) * inv_det;

        if (w2 < 0.0f || (w1 + w2) > 1.0f)
        {
            return false;
        }

        //Fill the intersection record.
        auto w0 = (1.0f - w1 - w2);
        intersection.normal = glm::normalize(w0 * m_normal0 + w1 * m_normal1 + w2 * m_normal2);
        intersection.tex_coord = w0 * m_texcoord0 + w1 * m_texcoord1 + w2 * m_texcoord2;
        intersection.distance = glm::dot(m_edge2, qvec) * inv_det;

        return true;
    }

    __device__ inline float intersectShadowRay(const Ray& ray) const
    {
        //Möller-Trumbore algorithm
        auto pvec = glm::cross(ray.get_direction(), m_edge2);
        auto inv_det = 1.0f / glm::dot(m_edge1, pvec);

        auto tvec = ray.get_origin() - m_v0;
        auto w1 = glm::dot(tvec, pvec) * inv_det;

        if (w1 < 0.0f || w1 > 1.0f)
        {
            return -1.0f;
        }

        auto qvec = glm::cross(tvec, m_edge1);
        auto w2 = glm::dot(ray.get_direction(), qvec) * inv_det;

        if (w2 < 0.0f || (w1 + w2) > 1.0f)
        {
            return -1.0f;
        }

        return glm::dot(m_edge2, qvec) * inv_det;
    }

private:
    glm::vec3 m_v0, m_edge1, m_edge2;
    glm::vec3 m_normal0, m_normal1, m_normal2;
    glm::vec2 m_texcoord0, m_texcoord1, m_texcoord2;
};