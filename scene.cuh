//raytracer.mustafaisik.net//

#pragma once

#include "bbox.cuh"
#include "bvh.cuh"
#include "jittered_sampler.cuh"

#include <glm/glm.hpp>

#include <vector>

#include "cuda_runtime.h"

class Material;
class Mesh;
class IntensityLight;
class RectangleLight;
class Ray;
class Sphere;
class Texture;
class Triangle;
struct Intersection;

class Scene
{
public:
    BBox m_bbox;
    std::pair<Sphere*, int> spheres;
    std::pair<Mesh*, int> meshes;
    std::pair<IntensityLight*, int> intensity_lights;
    std::pair<RectangleLight*, int> rectangle_lights;
    std::pair<Material*, int> materials;
    std::pair<Texture*, int> textures;
    std::pair<BVH::BVHNode*, int> nodes;
    glm::vec3 background_color;
    JitteredSampler jittered_sampler;
    float shadow_ray_epsilon;

public:
    __host__ Scene();
    __host__ ~Scene();
    __host__ Scene(const Scene&) = delete;
    __host__ Scene(Scene&&) = delete;
    __host__ Scene& operator=(const Scene&) = delete;
    __host__ Scene& operator=(Scene&&) = delete;

    __host__ void addSpheres(const std::vector<Sphere>& p_spheres);
    __host__ void addMeshes(const std::vector<Mesh>& p_meshes);
    __host__ void addIntensityLights(const std::vector<IntensityLight>& p_lights);
    __host__ void addRectangleLights(const std::vector<RectangleLight>& p_lights);
    __host__ void addTextures(const std::vector<Texture>& p_textures);
    __host__ void addMaterials(const std::vector<Material>& p_materials);

    __device__ bool intersect(const Ray& ray, Intersection& intersection, float max_distance) const;
    __device__ bool intersectShadowRay(const Ray& ray, float max_distance) const;
};