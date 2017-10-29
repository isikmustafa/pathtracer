//raytracer.mustafaisik.net//

#pragma once

#include "ray.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"

struct BBox
{
public:
    glm::vec3 min;
    glm::vec3 max;

public:
    __host__ BBox();
    __host__ BBox(const glm::vec3& min, const glm::vec3& max);

    __host__ void extend(const glm::vec3& point);
    __host__ void extend(const BBox& bbox);
    __host__ float getSurfaceArea() const;

    __device__ inline float intersect(const glm::vec3& origin, const glm::vec3& inv_dir) const
    {
        auto t0 = (min - origin) * inv_dir;
        auto t1 = (max - origin) * inv_dir;

        auto tmin = fminf(t0.x, t1.x);
        auto tmax = fmaxf(t0.x, t1.x);
        if (tmax < tmin)
        {
            return false;
        }

        tmin = fmaxf(tmin, fminf(t0.y, t1.y));
        tmax = fminf(tmax, fmaxf(t0.y, t1.y));
        if (tmax < tmin)
        {
            return false;
        }

        tmin = fmaxf(tmin, fminf(t0.z, t1.z));
        tmax = fminf(tmax, fmaxf(t0.z, t1.z));
        if (tmax < tmin)
        {
            return false;
        }

        return tmin > 0.0f ? tmin : tmax;
    }
};