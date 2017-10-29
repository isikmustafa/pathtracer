//raytracer.mustafaisik.net//

#include "scene.cuh"
#include "memory_handler.cuh"
#include "intersection.cuh"
#include "material.cuh"
#include "mesh.cuh"
#include "intensity_light.cuh"
#include "rectangle_light.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "texture.cuh"
#include "triangle.cuh"

__host__ Scene::Scene()
    : spheres(nullptr, 0)
    , meshes(nullptr, 0)
    , intensity_lights(nullptr, 0)
    , rectangle_lights(nullptr, 0)
    , materials(nullptr, 0)
    , nodes(nullptr, 0)
    , background_color(0.0f, 0.0f, 0.0f)
    , jittered_sampler(1)
    , shadow_ray_epsilon(0.001f)
{}

__host__ Scene::~Scene()
{}

__host__ void Scene::addSpheres(const std::vector<Sphere>& p_spheres)
{
    //This function is effective only when it is first called.
    if (!spheres.first)
    {
        auto size = p_spheres.size();
        if (size)
        {
            spheres.second = size;
            size_t data_size = spheres.second * sizeof(Sphere);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_spheres.begin()._Ptr));
            spheres.first = static_cast<Sphere*>(memory.pointer);
        }
    }
}

__host__ void Scene::addMeshes(const std::vector<Mesh>& p_meshes)
{
    //This function is effective only when it is first called.
    if (!nodes.first)
    {
        std::vector<BVH::BVHNode> p_nodes;
        m_bbox = BVH::buildSceneBvh(p_meshes, p_nodes);

        auto size = p_nodes.size();
        if (size)
        {
            nodes.second = size;
            size_t data_size = nodes.second * sizeof(BVH::BVHNode);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_nodes.begin()._Ptr));
            nodes.first = static_cast<BVH::BVHNode*>(memory.pointer);
        }
    }

    if (!meshes.first)
    {
        auto size = p_meshes.size();
        if (size)
        {
            meshes.second = size;
            size_t data_size = meshes.second * sizeof(Mesh);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_meshes.begin()._Ptr));
            meshes.first = static_cast<Mesh*>(memory.pointer);
        }
    }
}

__host__ void Scene::addIntensityLights(const std::vector<IntensityLight>& p_intensity_lights)
{
    //This function is effective only when it is first called.
    if (!intensity_lights.first)
    {
        auto size = p_intensity_lights.size();
        if (size)
        {
            intensity_lights.second = size;
            size_t data_size = intensity_lights.second * sizeof(IntensityLight);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_intensity_lights.begin()._Ptr));
            intensity_lights.first = static_cast<IntensityLight*>(memory.pointer);
        }
    }
}

__host__ void Scene::addRectangleLights(const std::vector<RectangleLight>& p_rectangle_lights)
{
    //This function is effective only when it is first called.
    if (!rectangle_lights.first)
    {
        auto size = p_rectangle_lights.size();
        if (size)
        {
            rectangle_lights.second = size;
            size_t data_size = rectangle_lights.second * sizeof(RectangleLight);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_rectangle_lights.begin()._Ptr));
            rectangle_lights.first = static_cast<RectangleLight*>(memory.pointer);
        }
    }
}

__host__ void Scene::addTextures(const std::vector<Texture>& p_textures)
{
    //This function is effective only when it is first called.
    if (!textures.first)
    {
        auto size = p_textures.size();
        if (size)
        {
            textures.second = size;
            size_t data_size = textures.second * sizeof(Texture);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_textures.begin()._Ptr));
            textures.first = static_cast<Texture*>(memory.pointer);
        }
    }
}

__host__ void Scene::addMaterials(const std::vector<Material>& p_materials)
{
    //This function is effective only when it is first called.
    if (!materials.first)
    {
        auto size = p_materials.size();
        if (size)
        {
            materials.second = size;
            size_t data_size = materials.second * sizeof(Material);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, p_materials.begin()._Ptr));
            materials.first = static_cast<Material*>(memory.pointer);
        }
    }
}

__device__ bool Scene::intersect(const Ray& ray, Intersection& intersection, float max_distance) const
{
    //TODO: Use Scene BVH to travese.
    Intersection intersection_temp;
    float min_distance = max_distance;
    bool is_hit = false;

    for (int i = 0; i < rectangle_lights.second; ++i)
    {
        if (rectangle_lights.first[i].intersect(ray, intersection_temp, min_distance))
        {
            min_distance = intersection_temp.distance;
            intersection = intersection_temp;
            is_hit = true;
        }
    }

    for (int i = 0; i < spheres.second; ++i)
    {
        if (spheres.first[i].intersect(ray, intersection_temp, min_distance))
        {
            min_distance = intersection_temp.distance;
            intersection = intersection_temp;
            is_hit = true;
        }
    }

    for (int i = 0; i < meshes.second; ++i)
    {
        if (meshes.first[i].intersect(ray, intersection_temp, min_distance))
        {
            min_distance = intersection_temp.distance;
            intersection = intersection_temp;
            is_hit = true;
        }
    }

    if (is_hit)
    {
        intersection.point = ray.getPoint(intersection.distance);
    }

    return is_hit;
}

__device__ bool Scene::intersectShadowRay(const Ray& ray, float max_distance) const
{
    //TODO: Use Scene BVH to travese.
    for (int i = 0; i < rectangle_lights.second; ++i)
    {
        if (rectangle_lights.first[i].intersectShadowRay(ray, max_distance))
        {
            return true;
        }
    }

    for (int i = 0; i < spheres.second; ++i)
    {
        if (spheres.first[i].intersectShadowRay(ray, max_distance))
        {
            return true;
        }
    }

    for (int i = 0; i < meshes.second; ++i)
    {
        if (meshes.first[i].intersectShadowRay(ray, max_distance))
        {
            return true;
        }
    }

    return false;
}