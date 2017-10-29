//raytracer.mustafaisik.net//

#include "bbox.cuh"
#include "math_utils.cuh"

__host__ BBox::BBox()
    : min(math::float_max, math::float_max, math::float_max)
    , max(-math::float_max, -math::float_max, -math::float_max)
{}

__host__ BBox::BBox(const glm::vec3& p_min, const glm::vec3& p_max)
    : min(p_min)
    , max(p_max)
{}

__host__ void BBox::extend(const glm::vec3& point)
{
    min = glm::min(min, point);
    max = glm::max(max, point);
}

__host__ void BBox::extend(const BBox& bbox)
{
    min = glm::min(min, bbox.min);
    max = glm::max(max, bbox.max);
}

__host__ float BBox::getSurfaceArea() const
{
    auto edges = max - min;
    return 2 * (edges.x * edges.y + edges.x * edges.z + edges.y * edges.z);
}