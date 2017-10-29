//raytracer.mustafaisik.net//

#include "sphere.cuh"

__host__ Sphere::Sphere(const Instance& instance, unsigned int material_id)
    : m_instance(instance)
    , m_material_id(material_id)
{}