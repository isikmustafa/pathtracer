//raytracer.mustafaisik.net//

#pragma once

#include "instance.cuh"
#include "intersection.cuh"
#include "math_utils.cuh"
#include "ray.cuh"
#include "texture.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

//Sphere has radius of 1 and is centered at (0,0,0).
class Sphere
{
public:
    __host__ Sphere(const Instance& instance, unsigned int material_id);

    __device__ inline bool intersect(const Ray& ray, Intersection& intersection, float max_distance) const
    {
        //Analytic solution.
        auto transformed_ray = m_instance.rayToObjectSpace(ray);
        auto a = glm::dot(transformed_ray.get_direction(), transformed_ray.get_direction());
        auto b = 2 * glm::dot(transformed_ray.get_direction(), transformed_ray.get_origin());
        auto c = glm::dot(transformed_ray.get_origin(), transformed_ray.get_origin()) - 1;

        auto discriminant = b * b - 4 * a * c;
        if (discriminant < 0.0f)
        {
            return false;
        }
        else
        {
            auto delta = sqrt(discriminant);
            auto div = 0.5f / a; //To prevent cancelling out.
            intersection.distance = -div * b - div * delta;

            if (intersection.distance < 0.0f)
            {
                intersection.distance = -div * b + div * delta;
            }
        }

        if (intersection.distance > 0.0f && intersection.distance < max_distance)
        {
            auto intersection_point = transformed_ray.getPoint(intersection.distance);
            intersection.normal = m_instance.normalToWorldSpace(intersection_point);
            intersection.light_emission = glm::vec3(-1.0f, -1.0f, -1.0f);
            auto phi = atan2f(intersection_point.z, intersection_point.x);
            auto theta = acosf(intersection_point.y);
            auto u = phi * math::inv_two_pi;
            auto v = theta * math::inv_pi;
            intersection.tex_coord = glm::vec2(u < 0.0f ? -u : 1.0f - u, v);
            intersection.material_id = m_material_id;

            return true;
        }

        return false;
    }

    __device__ inline bool intersectShadowRay(const Ray& ray, float max_distance) const
    {
        //Analytic solution.
        auto transformed_ray = m_instance.rayToObjectSpace(ray);
        auto a = glm::dot(transformed_ray.get_direction(), transformed_ray.get_direction());
        auto b = 2 * glm::dot(transformed_ray.get_direction(), transformed_ray.get_origin());
        auto c = glm::dot(transformed_ray.get_origin(), transformed_ray.get_origin()) - 1;

        auto distance = 0.0f;
        auto discriminant = b * b - 4 * a * c;
        if (discriminant < 0.0f)
        {
            return false;
        }
        else
        {
            auto delta = sqrt(discriminant);
            auto div = 0.5f / a; //To prevent cancelling out.
            distance = -div * b - div * delta;

            if (distance < 0.0f)
            {
                distance = -div * b + div * delta;
            }
        }

        return (distance > 0.0f && distance < max_distance);
    }

private:
    Instance m_instance;
    unsigned int m_material_id;
};