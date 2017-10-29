//raytracer.mustafaisik.net//

#pragma once

#include "bbox.cuh"
#include "bvh.cuh"
#include "instance.cuh"
#include "texture.cuh"
#include "triangle.cuh"

#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Instance;
class Ray;
class Triangle;
struct Intersection;

class Mesh
{
public:
    __host__ Mesh();
    __host__ Mesh(const Instance& instance, const BBox& bbox, unsigned int material_id);

    __host__ void createBaseMesh(const std::vector<Triangle>& mesh_triangles);
    __host__ void createInstanceMesh(const Mesh& base_mesh);

    __device__ inline bool intersect(const Ray& ray, Intersection& intersection, float max_distance) const
    {
        auto min_distance = max_distance;

        auto ray_origin = ray.get_origin();
        auto ray_inv_dir = 1.0f / ray.get_direction();
        auto result = m_bbox.intersect(ray_origin, ray_inv_dir);
        if (result <= 0.0f || result >= min_distance)
        {
            return false;
        }

        int triangle_index = -1;
        BVH::BVHNode* traversal_stack[32];
        int traversal_stack_top = -1;
        BVH::BVHNode* intersected_ptr = nullptr;

        //inner node
        if (m_nodes.first[0].left_node)
        {
            traversal_stack[++traversal_stack_top] = &m_nodes.first[0];;
        }
        //leaf node
        else
        {
            intersected_ptr = &m_nodes.first[0];
        }

        auto t_ray = m_instance.rayToObjectSpace(ray);
        ray_origin = t_ray.get_origin();
        ray_inv_dir = 1.0f / t_ray.get_direction();
        while (traversal_stack_top >= 0)
        {
            auto node = traversal_stack[traversal_stack_top--];

            result = node->left_bbox.intersect(ray_origin, ray_inv_dir);
            if (result > 0.0f && result < min_distance)
            {
                //inner node
                if (m_nodes.first[node->left_node].left_node)
                {
                    traversal_stack[++traversal_stack_top] = &m_nodes.first[node->left_node];
                }
                //leaf node
                else
                {
                    if (intersected_ptr)
                    {
                        for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                        {
                            auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                            if (result > 0.0f && result < min_distance)
                            {
                                min_distance = result;
                                triangle_index = i;
                            }
                        }
                    }
                    intersected_ptr = &m_nodes.first[node->left_node];
                }
            }
            result = node->right_bbox.intersect(ray_origin, ray_inv_dir);
            if (result > 0.0f && result < min_distance)
            {
                //inner node
                if (m_nodes.first[node->right_node].right_node)
                {
                    traversal_stack[++traversal_stack_top] = &m_nodes.first[node->right_node];
                }
                //leaf node
                else
                {
                    if (intersected_ptr)
                    {
                        for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                        {
                            auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                            if (result > 0.0f && result < min_distance)
                            {
                                min_distance = result;
                                triangle_index = i;
                            }
                        }
                    }
                    intersected_ptr = &m_nodes.first[node->right_node];
                }
            }

            if (__all(intersected_ptr != nullptr))
            {
                for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                {
                    auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                    if (result > 0.0f && result < min_distance)
                    {
                        min_distance = result;
                        triangle_index = i;
                    }
                }

                intersected_ptr = nullptr;
            }
        }

        if (intersected_ptr)
        {
            for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
            {
                auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                if (result > 0.0f && result < min_distance)
                {
                    min_distance = result;
                    triangle_index = i;
                }
            }
        }

        if (triangle_index >= 0)
        {
            m_triangles.first[triangle_index].intersect(t_ray, intersection);
            intersection.normal = m_instance.normalToWorldSpace(intersection.normal);
            intersection.light_emission = glm::vec3(-1.0f, -1.0f, -1.0f);
            intersection.material_id = m_material_id;

            return true;
        }

        return false;
    }

    __device__ inline bool intersectShadowRay(const Ray& ray, float max_distance) const
    {
        auto ray_origin = ray.get_origin();
        auto ray_inv_dir = 1.0f / ray.get_direction();
        auto result = m_bbox.intersect(ray_origin, ray_inv_dir);
        if (result <= 0.0f || result >= max_distance)
        {
            return false;
        }

        BVH::BVHNode* traversal_stack[32];
        int traversal_stack_top = -1;
        BVH::BVHNode* intersected_ptr = nullptr;

        //inner node
        if (m_nodes.first[0].left_node)
        {
            traversal_stack[++traversal_stack_top] = &m_nodes.first[0];
        }
        //leaf node
        else
        {
            intersected_ptr = &m_nodes.first[0];
        }

        auto t_ray = m_instance.rayToObjectSpace(ray);
        ray_origin = t_ray.get_origin();
        ray_inv_dir = 1.0f / t_ray.get_direction();
        while (traversal_stack_top >= 0)
        {
            auto node = traversal_stack[traversal_stack_top--];

            result = node->left_bbox.intersect(ray_origin, ray_inv_dir);
            if (result > 0.0f && result < max_distance)
            {
                //inner node
                if (m_nodes.first[node->left_node].left_node)
                {
                    traversal_stack[++traversal_stack_top] = &m_nodes.first[node->left_node];
                }
                //leaf node
                else
                {
                    if (intersected_ptr)
                    {
                        for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                        {
                            auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                            if (result > 0.0f && result < max_distance)
                            {
                                return true;
                            }
                        }
                    }
                    intersected_ptr = &m_nodes.first[node->left_node];
                }
            }
            result = node->right_bbox.intersect(ray_origin, ray_inv_dir);
            if (result > 0.0f && result < max_distance)
            {
                //inner node
                if (m_nodes.first[node->right_node].right_node)
                {
                    traversal_stack[++traversal_stack_top] = &m_nodes.first[node->right_node];
                }
                //leaf node
                else
                {
                    if (intersected_ptr)
                    {
                        for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                        {
                            auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                            if (result > 0.0f && result < max_distance)
                            {
                                return true;
                            }
                        }
                    }
                    intersected_ptr = &m_nodes.first[node->right_node];
                }
            }

            if (__all(intersected_ptr != nullptr))
            {
                for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
                {
                    auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                    if (result > 0.0f && result < max_distance)
                    {
                        return true;
                    }
                }

                intersected_ptr = nullptr;
            }
        }

        if (intersected_ptr)
        {
            for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
            {
                auto result = m_triangles.first[i].intersectShadowRay(t_ray);
                if (result > 0.0f && result < max_distance)
                {
                    return true;
                }
            }
        }

        return false;
    }

    //Getters
    __host__ const BBox& get_bbox() const
    {
        return m_bbox;
    }

private:
    Instance m_instance;
    BBox m_bbox;
    std::pair<Triangle*, int> m_triangles;
    std::pair<BVH::BVHNode*, int> m_nodes;
    unsigned int m_material_id;
};