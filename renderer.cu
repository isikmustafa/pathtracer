//raytracer.mustafaisik.net//

#include "renderer.cuh"
#include "camera.cuh"
#include "cuda_event.cuh"
#include "memory_handler.cuh"
#include "intersection.cuh"
#include "material.cuh"
#include "intensity_light.cuh"
#include "rectangle_light.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "texture.cuh"
#include "triangle.cuh"
#include "cuda_utils.cuh"

#include <iostream>

#include <glm/glm.hpp>
#include <FreeImage/FreeImage.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void pathTrace(Image* image, Camera* camera, Scene* scene, float* accum, int sample_count)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image->get_width() || j >= image->get_height())
    {
        return;
    }

    int pixel_index = i + j * image->get_width();

    //Cantor pairing function for unique seed.
    unsigned long long ha = pixel_index;
    unsigned long long hb = sample_count;
    unsigned long long seed = (ha + hb) * (ha + hb + 1) / 2 + ha;

    curandState rand_state;
    curand_init(seed, 0, 0, &rand_state);
    glm::vec3 final_color(0.0f, 0.0f, 0.0f);

    auto intensity_light_count = scene->intensity_lights.second;
    auto rectangle_light_count = scene->rectangle_lights.second;

    PathInfo path_info(camera->castPrimayRay(i, j, scene->jittered_sampler.getSample(rand_state, sample_count)), glm::vec3(1.0f, 1.0f, 1.0f), 0);
    bool explicit_light_sample = false;

    while (true)
    {
        //Russian roulette
        auto weight = fmaxf(fmaxf(path_info.weight.x, path_info.weight.y), path_info.weight.z);
        weight = weight > 1.0f ? 1.0f : weight;
        if (path_info.depth > 3 && !path_info.in_medium)
        {
            if (1.0f - curand_uniform(&rand_state) < weight)
            {
                path_info.weight /= weight;
            }
            else
            {
                break;
            }
        }

        Intersection intersection;
        auto result = scene->intersect(path_info.ray, intersection, math::float_max);

        if (!result)
        {
            final_color += (scene->background_color * path_info.weight);
            break;
        }

        if (intersection.light_emission.x >= 0.0f)
        {
            //If we did not explicitly sample the light sources in the previous bounce.
            if (!explicit_light_sample)
            {
                final_color += (intersection.light_emission * path_info.weight);
            }
            break;
        }

        auto& material = scene->materials.first[intersection.material_id];
        auto material_type = material.get_type();
        if (material.get_type() == Material::EMISSIVE)
        {
            final_color += material.computeEmissiveRadiance(intersection, -path_info.ray.get_direction()) * path_info.weight;
            break;
        }
        else if (material_type == Material::LAMBERTIAN)
        {
            glm::vec3 outgoing_radiance(0.0f, 0.0f, 0.0f);
            explicit_light_sample = true;

            //Sample all intensity light sources.
            for (int i = 0; i < intensity_light_count; ++i)
            {
                auto& light = scene->intensity_lights.first[i];
                auto light_hit = light.getLightHit(intersection.point);

                //Make sure that the shadow ray and the primary ray leave the surface on the same side.
                if (glm::dot(light_hit.to_light, intersection.normal) < math::float_epsilon)
                {
                    continue;
                }

                Ray shadow_ray(intersection.point + scene->shadow_ray_epsilon * light_hit.to_light, light_hit.to_light);
                result = scene->intersectShadowRay(shadow_ray, light_hit.distance);

                if (!result)
                {
                    auto incoming_radiance = light.computeRadiance(light_hit);
                    outgoing_radiance +=
                        incoming_radiance * material.computeDiffuseReflectance(intersection) * fmaxf(glm::dot(light_hit.to_light, intersection.normal), 0.0f);
                }
            }

            //Sample all rectangle light sources.
            for (int i = 0; i < rectangle_light_count; ++i)
            {
                auto& light = scene->rectangle_lights.first[i];
                auto light_hit = light.getLightHit(intersection.point, glm::vec2(curand_uniform(&rand_state), curand_uniform(&rand_state)));

                //Make sure that the shadow ray and the primary ray leave the surface on the same side.
                if (glm::dot(light_hit.to_light, intersection.normal) < math::float_epsilon)
                {
                    continue;
                }

                Ray shadow_ray(intersection.point + scene->shadow_ray_epsilon * light_hit.to_light, light_hit.to_light);
                result = scene->intersectShadowRay(shadow_ray, light_hit.distance - 0.01f);

                if (!result)
                {
                    auto incoming_radiance = light.computeRadiance(light_hit);
                    outgoing_radiance +=
                        incoming_radiance * material.computeDiffuseReflectance(intersection) * fmaxf(glm::dot(light_hit.to_light, intersection.normal), 0.0f);
                }
            }

            final_color += outgoing_radiance * path_info.weight;

            //Sample new direction for the indirect part.
            //Build the orthonormal basis.
            auto u = glm::normalize(glm::cross((fabsf(intersection.normal.x) > 0.1f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)), intersection.normal));
            auto v = glm::cross(intersection.normal, u);

            //Cosine-weighted sampling.
            auto phi = 2 * math::pi * curand_uniform(&rand_state);
            auto u2 = curand_uniform(&rand_state);

            //Formula is reduced to an efficient version. It may not seem neat.
            auto sampled_direction = glm::normalize(sqrtf(u2) * (cosf(phi) * u + sinf(phi) * v) + sqrtf(1.0f - u2) * intersection.normal);
            path_info.weight *= material.computeDiffuseReflectance(intersection) * math::pi;
            path_info.ray = Ray(intersection.point + scene->shadow_ray_epsilon * sampled_direction, sampled_direction);
        }
        else if (material_type == Material::PERFECT_SPECULAR)
        {
            explicit_light_sample = false;
            material.generateMirrorRay(intersection, path_info);
        }
        else if (material_type == Material::PERFECT_REFRACTIVE)
        {
            explicit_light_sample = false;
            material.generateGlassRay(intersection, path_info, rand_state);
        }
        else if (material_type == Material::TRANSLUCENT)
        {
            explicit_light_sample = false;
            material.generateMediumRay(intersection, path_info, rand_state);
        }

        ++path_info.depth;
    }

    if (sample_count == 0)
    {
        accum[pixel_index * 3] = final_color.x;
        accum[pixel_index * 3 + 1] = final_color.y;
        accum[pixel_index * 3 + 2] = final_color.z;
    }
    else
    {
        accum[pixel_index * 3] += final_color.x;
        accum[pixel_index * 3 + 1] += final_color.y;
        accum[pixel_index * 3 + 2] += final_color.z;
    }

    final_color = glm::vec3(accum[pixel_index * 3], accum[pixel_index * 3 + 1], accum[pixel_index * 3 + 2]) / static_cast<float>(sample_count + 1);
    image->setPixel(pixel_index, glm::clamp(final_color, 0.0f, 255.0f));
}

Renderer::Renderer(int image_width, int image_height)
    : m_image(image_width, image_height)
    , m_image_width(image_width)
    , m_image_height(image_height)
{}

const Image& Renderer::render(const Camera& camera, const Scene& scene, int sample_count) const
{
    static Image* device_image = static_cast<Image*>(MemoryHandler::Handler().allocateOnDevice(sizeof(Image), Memory(Memory::HOST, (void*)(&m_image))).pointer);
    static Camera* device_camera = static_cast<Camera*>(MemoryHandler::Handler().allocateOnDevice(sizeof(Camera)).pointer);
    MemoryHandler::copy(Memory(Memory::DEVICE, device_camera), Memory(Memory::HOST, (void*)(&camera)), sizeof(Camera));
    static Scene* device_scene = static_cast<Scene*>(MemoryHandler::Handler().allocateOnDevice(sizeof(Scene), Memory(Memory::HOST, (void*)(&scene))).pointer);
    static float* accum = static_cast<float*>(MemoryHandler::Handler().allocateOnDevice(sizeof(float) * 3 * m_image_width * m_image_height).pointer);

    /*KERNEL*/
    static CudaEvent start, end;

    dim3 threads(8, 8);
    int block_x = std::ceil(static_cast<float>(m_image_width) / threads.x);
    int block_y = std::ceil(static_cast<float>(m_image_height) / threads.y);
    dim3 blocks(block_x, block_y);
    std::cout << "Kernel execution is started" << std::endl;

    start.record();
    pathTrace << <blocks, threads >> > (device_image, device_camera, device_scene, accum, sample_count);

    end.record();
    end.synchronize();
    std::cout << "Kernel execution time: " << CudaEvent::calculateElapsedTime(start, end) << " ms" << std::endl;
    /*KERNEL*/

    return m_image;
}