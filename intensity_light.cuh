//raytracer.mustafaisik.net//

#pragma once

#include <glm/glm.hpp>

#include "cuda_runtime.h"

class IntensityLight
{
public:
    struct LightHit
    {
        glm::vec3 to_light;
        float distance;
        float attenuation;
    };

    enum Type
    {
        POINT,
        SPOT
    };

public:
    __host__ IntensityLight(const glm::vec3& position, const glm::vec3& intensity)
        : m_position(position)
        , m_intensity(intensity)
        , m_type(POINT)
    {}

    __host__ IntensityLight(const glm::vec3& position, const glm::vec3& intensity, const glm::vec3& direction, float falloff_angle, float cutoff_angle)
        : m_position(position)
        , m_intensity(intensity)
        , m_direction(glm::normalize(direction))
        , m_falloff_cosine(glm::cos(glm::radians(falloff_angle)))
        , m_cutoff_cosine(glm::cos(glm::radians(cutoff_angle)))
        , m_type(SPOT)
    {}

    __device__ inline LightHit getLightHit(const glm::vec3& point) const
    {
        LightHit light_hit;
        light_hit.to_light = m_position - point;
        light_hit.distance = glm::length(light_hit.to_light);
        light_hit.to_light /= light_hit.distance;
        if (m_type == POINT)
        {
            light_hit.attenuation = 1.0f;
        }
        else if (m_type == SPOT)
        {
            light_hit.attenuation = glm::clamp((glm::dot(-light_hit.to_light, m_direction) - m_cutoff_cosine) / (m_falloff_cosine - m_cutoff_cosine), 0.0f, 1.0f);
        }

        return light_hit;
    }

    __device__ inline glm::vec3 computeRadiance(const LightHit& light_hit) const
    {
        return (m_intensity * light_hit.attenuation) / (light_hit.distance * light_hit.distance);
    }

private:
    glm::vec3 m_position;
    glm::vec3 m_intensity;
    glm::vec3 m_direction;
    float m_falloff_cosine;
    float m_cutoff_cosine;
    Type m_type;
};