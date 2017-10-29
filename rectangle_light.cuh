//raytracer.mustafaisik.net//

#pragma once

#include "intersection.cuh"
#include "ray.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

class RectangleLight
{
public:
    struct LightHit
    {
        glm::vec3 to_light;
        float distance;
    };

public:
    __host__ RectangleLight(const glm::vec3& radiance, const glm::vec3& position, const glm::vec3& edge_u, const glm::vec3& edge_v)
        : m_radiance(radiance)
        , m_position(position)
        , m_edge_u(edge_u)
        , m_edge_v(edge_v)
        , m_normal(glm::normalize(glm::cross(edge_u, edge_v)))
        , m_area(glm::length(glm::cross(edge_u, edge_v)))
    {}

    __device__ bool intersect(const Ray& ray, Intersection& intersection, float max_distance) const
    {
        auto x = glm::dot(ray.get_direction(), m_normal);

        if (fabsf(x) <= math::float_epsilon)
        {
            return false;
        }

        intersection.distance = (glm::dot(m_position, m_normal) - glm::dot(ray.get_origin(), m_normal)) / x;

        if (intersection.distance > 0.0f && intersection.distance < max_distance)
        {
            auto local_position = ray.getPoint(intersection.distance) - m_position;

            auto u = glm::dot(glm::normalize(m_edge_u), local_position);
            auto v = glm::dot(glm::normalize(m_edge_v), local_position);

            if (u >= 0 && u * u <= dot(m_edge_u, m_edge_u) && v >= 0 && v * v <= dot(m_edge_v, m_edge_v))
            {
                intersection.light_emission = m_radiance * fmaxf(glm::dot(glm::normalize(ray.get_origin() - ray.getPoint(intersection.distance)), m_normal), 0.0f) * m_area;
                return true;
            }
        }

        return false;
    }

    __device__ bool intersectShadowRay(const Ray& ray, float max_distance) const
    {
        auto x = glm::dot(ray.get_direction(), m_normal);

        if (fabsf(x) <= math::float_epsilon)
        {
            return false;
        }

        auto distance = (glm::dot(m_position, m_normal) - glm::dot(ray.get_origin(), m_normal)) / x;

        if (distance > 0.0f && distance < max_distance)
        {
            auto local_position = ray.getPoint(distance) - m_position;

            auto u = glm::dot(glm::normalize(m_edge_u), local_position);
            auto v = glm::dot(glm::normalize(m_edge_v), local_position);

            if (u >= 0 && u * u <= dot(m_edge_u, m_edge_u) && v >= 0 && v * v <= dot(m_edge_v, m_edge_v))
            {
                return true;
            }
        }

        return false;
    }

    __device__ inline LightHit getLightHit(const glm::vec3& point, const glm::vec2& random_sample) const
    {
        LightHit light_hit;
        light_hit.to_light = m_position + m_edge_u * random_sample.x + m_edge_v * random_sample.y - point;
        light_hit.distance = glm::length(light_hit.to_light);
        light_hit.to_light /= light_hit.distance;

        return light_hit;
    }

    __device__ inline glm::vec3 computeRadiance(const LightHit& light_hit) const
    {
        return m_radiance * fmaxf(glm::dot(-light_hit.to_light, m_normal), 0.0f) * m_area;
    }

private:
    glm::vec3 m_radiance;
    glm::vec3 m_position;
    glm::vec3 m_edge_u;
    glm::vec3 m_edge_v;
    glm::vec3 m_normal;
    float m_area;
};