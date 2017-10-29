//raytracer.mustafaisik.net//

#pragma once

#include "math_utils.cuh"
#include "ray.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

//All the computations in this class(and of course the entire project) are performed using right-handed coordinate system.
class Camera
{
public:
    __host__ Camera(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& world_up,
        const glm::vec4& screen_coordinates, const glm::ivec2& screen_resolution,
        float aperture_radius, float focus_distance);

    __host__ void move(float right_disp, float forward_disp);
    __host__ void rotate(float radian_world_up, float radian_right);

    __device__ inline Ray castPrimayRay(float x, float y) const
    {
        x = m_coefficients.x * x + m_base_pixels.x;
        y = m_coefficients.y * y + m_base_pixels.y;

        return Ray(m_position, glm::normalize(-m_near_distance * m_forward + x * m_right - y * m_up));
    }
    __device__ inline Ray castPrimayRay(float x, float y, const glm::vec2& jittered_sample) const
    {
        x = m_coefficients.x * x + (m_coefficients.x * (jittered_sample.x - 0.5f) + m_base_pixels.x);
        y = m_coefficients.y * y + (m_coefficients.y * (jittered_sample.y - 0.5f) + m_base_pixels.y);
        if (m_aperture_radius == 0.0f)
        {
            return Ray(m_position, glm::normalize(-m_near_distance * m_forward + x * m_right - y * m_up));
        }
        else
        {
            //The point that the center ray is passing through on the focal plane.
            auto p = m_position + (-m_near_distance * m_forward + x * m_right - y * m_up) * (m_focus_distance / m_near_distance);

            //Pick a sample on the disk of radius m_aperture_radius.
            auto phi = math::two_pi * jittered_sample.x;
            auto r = sqrtf(jittered_sample.y) * m_aperture_radius;
            auto c = m_position + r * cosf(phi) * m_right + r * sinf(phi) * m_up;

            return Ray(c, glm::normalize(p - c));
        }
    }

private:
    glm::vec3 m_position;
    glm::vec3 m_right;
    glm::vec3 m_up;
    glm::vec3 m_forward;
    glm::vec3 m_world_up;
    glm::vec2 m_coefficients;
    glm::vec2 m_base_pixels;
    float m_near_distance;
    float m_focus_distance;
    float m_aperture_radius;

private:
    __host__ void buildView();
};