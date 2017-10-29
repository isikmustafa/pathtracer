//raytracer.mustafaisik.net//

#pragma once

#include "texture.cuh"

#include <glm/glm.hpp>

struct Intersection
{
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 light_emission;
    glm::vec2 tex_coord;
    float distance;
    unsigned int material_id;
};