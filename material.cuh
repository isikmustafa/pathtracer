//raytracer.mustafaisik.net//

#pragma once

#include "intersection.cuh"
#include "math_utils.cuh"
#include "texture.cuh"

#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "curand_kernel.h"

struct PathInfo
{
    Ray ray;
    glm::vec3 weight;
    int depth;
    bool in_medium;

    __device__ PathInfo()
        : ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f))
        , weight(glm::vec3(1.0f, 1.0f, 1.0f))
        , depth(0)
        , in_medium(false)
    {}

    __device__ PathInfo(const Ray& p_secondary_ray, const glm::vec3& p_weight, int p_depth)
        : ray(p_secondary_ray)
        , weight(p_weight)
        , depth(p_depth)
        , in_medium(false)
    {}
};

class Material
{
public:
    enum Type
    {
        EMISSIVE,
        LAMBERTIAN,
        PERFECT_SPECULAR,
        PERFECT_REFRACTIVE,
        TRANSLUCENT
    };

public:
    __host__ Material(const glm::vec3& emission, const glm::vec3& diffuse, const glm::vec3& glossy_specular,
        const glm::vec3& perfect_specular, const glm::vec3& tint_color, float tint_distance, float scattering_coefficient, float g,
        const Texture* diffuse_texture, float ior, Type material_type)
        : m_emission(emission)
        , m_diffuse(diffuse)
        , m_glossy_specular(glossy_specular)
        , m_perfect_specular(perfect_specular)
        , m_tint_color(tint_color)
        , m_tint_distance(tint_distance)
        , m_scattering_coeff(scattering_coefficient)
        , m_g(g)
        , m_diffuse_texture(diffuse_texture)
        , m_ior(ior)
        , m_type(material_type)
    {}

    __device__ inline glm::vec3 computeEmissiveRadiance(const Intersection& intersection, const glm::vec3& wo) const
    {
        //Emission
        return m_emission * fmaxf(glm::dot(wo, intersection.normal), 0.0f);
    }

    __device__ inline glm::vec3 computeDiffuseReflectance(const Intersection& intersection) const
    {
        //Diffuse
        if (m_diffuse_texture)
        {
            if (m_diffuse_texture->get_texture_type() == Texture::IMAGE)
            {
                return (m_diffuse + m_diffuse_texture->sampleImage(intersection.tex_coord)) * 0.5f * math::inv_pi;
            }
            else
            {
                return (m_diffuse + m_diffuse_texture->samplePerlin(intersection.point)) * 0.5f * math::inv_pi;
            }
        }
        return m_diffuse * math::inv_pi;
    }

    __device__ inline glm::vec3 computeGlossyReflectance(const Intersection& intersection, const glm::vec3& wi, const glm::vec3& wo) const
    {
        //Glossy Specular
        /*return m_glossy_specular * m_brdf.glossy(intersection, wi, wo, m_glossy_specular);

        if (m_type == MODIFIED_BLINNPHONG)
        {
            return glm::vec3(powf(fmaxf(glm::dot(glm::normalize(wo + wi), intersection.normal), 0.0f), m_exponent) * (m_exponent + 8) * math::inv_eight_pi);
        }
        else if (m_type == COOK_TORRANCE)
        {
            auto h = glm::normalize(wo + wi);
            auto cosi = fmaxf(glm::dot(wi, intersection.normal), 0.0f);
            auto cosv = fmaxf(glm::dot(wo, intersection.normal), 0.0f);
            auto cosh = fmaxf(glm::dot(h, intersection.normal), 0.0f);
            auto cosvh = fmaxf(glm::dot(wo, h), 0.0f);
            auto d = (m_exponent + 2) * math::inv_two_pi * powf(cosh, m_exponent);
            auto cosx = 1.0f - cosvh;
            auto f = glossy_specular + (1.0f - glossy_specular) * cosx * cosx * cosx * cosx * cosx;
            auto g = fminf(fminf(1.0f, 2.0f * cosh * cosv / cosvh), 2.0f * cosh * cosi / cosvh);
            return (d * f * g * 0.25f) / (cosv * cosi);
        }
        else
        {
            return glm::vec3(0.0f, 0.0f, 0.0f);
        }*/
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    __device__ inline void generateMirrorRay(const Intersection& intersection, PathInfo& path_info) const
    {
        //Perfect Specular
        auto reflected_dir = path_info.ray.get_direction() - 2 * glm::dot(intersection.normal, path_info.ray.get_direction()) * intersection.normal;

        path_info.ray = Ray(intersection.point + 0.001f * reflected_dir, reflected_dir);
        path_info.weight *= m_perfect_specular;
    }

    __device__ inline void generateGlassRay(const Intersection& intersection, PathInfo& path_info, curandState& rand_state) const
    {
        //Default values for inside the medium.
        auto outside = false;
        auto normal = intersection.normal;
        auto costheta_i = glm::dot(path_info.ray.get_direction(), normal);
        auto eta_ratio = m_ior;

        //Outside the medium.
        if (costheta_i < 0.0f)
        {
            outside = true;
            costheta_i = -costheta_i;
            eta_ratio = 1.0f / m_ior;
        }
        //Inside the medium.
        else
        {
            normal = -normal;

            auto distance = glm::length(intersection.point - path_info.ray.get_origin());

            //Beer's Law.
            auto extinction_coeff = -glm::log(m_tint_color) / m_tint_distance;
            auto beam_transmittance = glm::exp(-extinction_coeff * distance);
            path_info.weight *= beam_transmittance;
        }

        auto r_schlick = 1.1f;
        auto sintheta_t2 = eta_ratio * eta_ratio * (1.0f - costheta_i * costheta_i);
        auto costheta_t = 0.0f;
        //If not a total internal reflection.
        if (sintheta_t2 <= 1.0f)
        {
            costheta_t = sqrtf(1.0f - sintheta_t2);

            //Schlick's approximation.
            auto r0 = (1.0f - m_ior) / (1.0f + m_ior);
            r0 *= r0;

            auto cosx = 1.0f - costheta_t;
            if (outside)
            {
                cosx = 1.0f - costheta_i;
            }
            r_schlick = r0 + (1.0f - r0) * cosx * cosx * cosx * cosx * cosx;
        }

        //r_schlick should be initialized to a number greater than 1. For example, 1.1f above.
        //No matter what random value we choose, it is less than 1.
        //So, in case of total internal reflection, we will be choosing the right branch to take.
        //Refraction
        if (curand_uniform(&rand_state) > r_schlick)
        {
            //Snell's law
            auto refracted_dir = eta_ratio * path_info.ray.get_direction() + (eta_ratio * costheta_i - costheta_t) * normal;
            path_info.ray = Ray(intersection.point - 0.001f * normal, refracted_dir);
        }
        //Reflection
        else
        {
            auto reflected_dir = path_info.ray.get_direction() + 2 * costheta_i * normal;
            path_info.ray = Ray(intersection.point + 0.001f * normal, reflected_dir);
        }
    }

    __device__ inline void generateMediumRay(const Intersection& intersection, PathInfo& path_info, curandState& rand_state) const
    {
        //Perfect refractive object with subsurface scattering inside.
        //Default values for inside the medium.
        auto outside = false;
        auto normal = intersection.normal;
        auto costheta_i = glm::dot(path_info.ray.get_direction(), normal);
        auto eta_ratio = m_ior;
        path_info.in_medium = true;

        //Outside the medium.
        if (costheta_i < 0.0f)
        {
            outside = true;
            costheta_i = -costheta_i;
            eta_ratio = 1.0f / m_ior;
        }
        //Inside the medium.
        else
        {
            normal = -normal;

            auto tmax = glm::length(intersection.point - path_info.ray.get_origin());

            //Sample a distance on the ray.
            auto t = -(logf(curand_uniform(&rand_state)) / m_scattering_coeff);

            auto extinction_coeff = -glm::log(m_tint_color) / m_tint_distance + m_scattering_coeff;

            //Beer's Law.
            auto beam_transmittance = glm::exp(-extinction_coeff * fminf(t, tmax));
            auto pdf = expf(-m_scattering_coeff * fminf(t, tmax));
            path_info.weight *= (beam_transmittance / pdf);

            //Medium interaction.
            if (t < tmax)
            {
                //Sample the scattered direction with Henyey–Greenstein phase function.
                //Build the orthonormal basis.
                auto& ray_dir = path_info.ray.get_direction();
                auto u = glm::normalize(glm::cross((fabsf(ray_dir.x) > 0.1f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)), ray_dir));
                auto v = glm::cross(ray_dir, u);

                //Sample the direction with Henyey-Greenstein phase function.
                float cos_theta;
                if (fabsf(m_g) < 0.001f)
                {
                    cos_theta = 1.0f - 2.0f * curand_uniform(&rand_state);
                }
                else
                {
                    auto sqrd_term = (1.0f - m_g * m_g) / (1.0f - m_g + 2.0f * m_g * curand_uniform(&rand_state));
                    cos_theta = (1.0f + m_g * m_g - sqrd_term * sqrd_term) / (2.0f * m_g);
                }
                auto sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
                auto phi = math::two_pi * curand_uniform(&rand_state);

                auto sampled_direction = glm::normalize(sin_theta * (cosf(phi) * u + sinf(phi) * v) + cos_theta * ray_dir);
                auto origin = path_info.ray.getPoint(t);
                path_info.ray = Ray(origin, sampled_direction);

                return;
            }
            //Surface interaction.
            else
            {
                path_info.in_medium = false;
            }
        }

        auto r_schlick = 1.1f;
        auto sintheta_t2 = eta_ratio * eta_ratio * (1.0f - costheta_i * costheta_i);
        auto costheta_t = 0.0f;
        //If not a total internal reflection.
        if (sintheta_t2 <= 1.0f)
        {
            costheta_t = sqrtf(1.0f - sintheta_t2);

            //Schlick's approximation.
            auto r0 = (1.0f - m_ior) / (1.0f + m_ior);
            r0 *= r0;

            auto cosx = 1.0f - costheta_t;
            if (outside)
            {
                cosx = 1.0f - costheta_i;
            }
            r_schlick = r0 + (1.0f - r0) * cosx * cosx * cosx * cosx * cosx;
        }

        //r_schlick should be initialized to a number greater than 1. For example, 1.1f above.
        //No matter what random value we choose, it is less than 1.
        //So, in case of total internal reflection, we will be choosing the right branch to take.
        //Refraction
        if (curand_uniform(&rand_state) > r_schlick)
        {
            //Snell's law
            auto refracted_dir = eta_ratio * path_info.ray.get_direction() + (eta_ratio * costheta_i - costheta_t) * normal;
            path_info.ray = Ray(intersection.point - 0.001f * normal, refracted_dir);
        }
        //Reflection
        else
        {
            auto reflected_dir = path_info.ray.get_direction() + 2 * costheta_i * normal;
            path_info.ray = Ray(intersection.point + 0.001f * normal, reflected_dir);
        }
    }

    //Getters
    __device__ inline Type get_type() const
    {
        return m_type;
    }

private:
    glm::vec3 m_emission;
    glm::vec3 m_diffuse;
    glm::vec3 m_glossy_specular;
    glm::vec3 m_perfect_specular;
    glm::vec3 m_tint_color;
    float m_tint_distance;
    float m_scattering_coeff;
    float m_g;
    const Texture* m_diffuse_texture;
    float m_ior;
    Type m_type;
};