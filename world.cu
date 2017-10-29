//raytracer.mustafaisik.net//

#include "world.cuh"
#include "camera.cuh"
#include "instance.cuh"
#include "material.cuh"
#include "mesh.cuh"
#include "intensity_light.cuh"
#include "rectangle_light.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "system.cuh"
#include "timer.cuh"
#include "texture_manager.cuh"

#include <fstream>
#include <sstream>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "gl/tinyxml2.h"

using namespace tinyxml2;

__constant__ float gPerlinGradients[36] = { 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f,
1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, -1.0f,
0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, -1.0f };

__constant__ int gPerlinPermutation[512] = { 151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

World::World()
    : m_system(nullptr)
    , m_renderer(nullptr)
    , m_camera(nullptr)
    , m_scene(nullptr)
    , m_camera_speed(1)
{}

World::~World()
{}

void World::loadScene(const std::string& filepath)
{
    XMLDocument file;
    std::stringstream stream;

    auto res = file.LoadFile(filepath.c_str());
    if (res)
    {
        throw std::runtime_error("Error: Cannot load the xml file.");
    }

    XMLNode* root = file.FirstChild();
    if (!root)
    {
        throw std::runtime_error("Error: Root is not found.");
    }

    //Camera speed
    auto element = root->FirstChildElement("CameraSpeed");
    if (element)
    {
        stream << element->GetText() << std::endl;
        stream >> m_camera_speed;
        stream.clear();
    }
    else
    {
        m_camera_speed = 1.0;
    }

    //Image name
    element = root->FirstChildElement("ImageName");
    if (element)
    {
        stream << element->GetText() << std::endl;
        stream >> m_image_name;
        stream.clear();
    }
    else
    {
        throw std::runtime_error("Error: Image name is not specified.");
    }

    //Image resolution
    element = root->FirstChildElement("ImageResolution");
    if (element)
    {
        stream << element->GetText() << std::endl;
        stream >> m_screen_width >> m_screen_height;
        stream.clear();
    }
    else
    {
        throw std::runtime_error("Error: Image resolution is not specified.");
    }


    //Camera
    glm::vec3 position, gaze, up;
    glm::vec4 near_plane;
    float near_distance;
    float focus_distance;
    float aperture_radius;

    element = root->FirstChildElement("Camera");
    if (!element)
    {
        throw std::runtime_error("Error: Camera is not specified.");
    }

    //Camera-position
    auto child = element->FirstChildElement("Position");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        throw std::runtime_error("Error: Camera Position is not specified.");
    }

    //Camera-gaze
    child = element->FirstChildElement("Gaze");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        throw std::runtime_error("Error: Camera Gaze is not specified.");
    }

    //Camera-up
    child = element->FirstChildElement("Up");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        throw std::runtime_error("Error: Camera Up is not specified.");
    }

    //Camera-near plane
    child = element->FirstChildElement("NearPlane");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        throw std::runtime_error("Error: Camera NearPlane is not specified.");
    }

    //Camera-near distance
    child = element->FirstChildElement("NearDistance");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        stream << "1" << std::endl;
    }

    //Camera-focus distance
    child = element->FirstChildElement("FocusDistance");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        stream << "1" << std::endl;
    }

    //Camera-aperture radius
    child = element->FirstChildElement("ApertureRadius");
    if (child)
    {
        stream << child->GetText() << std::endl;
    }
    else
    {
        stream << "0" << std::endl;
    }

    stream >> position.x >> position.y >> position.z;
    stream >> gaze.x >> gaze.y >> gaze.z;
    stream >> up.x >> up.y >> up.z;
    stream >> near_plane.x >> near_plane.y >> near_plane.z >> near_plane.w;
    stream >> near_distance;
    stream >> focus_distance;
    stream >> aperture_radius;
    stream.clear();

    //Initialize the modules.
    m_renderer.reset(new Renderer(m_screen_width, m_screen_height));
    m_camera.reset(new Camera(position, glm::normalize(gaze) * near_distance, up, near_plane, glm::ivec2(m_screen_width, m_screen_height), aperture_radius, focus_distance));
    m_scene.reset(new Scene());

    //Some constants
    element = root->FirstChildElement("BackgroundColor");
    if (element)
    {
        stream << element->GetText() << std::endl;
        stream >> m_scene->background_color.x >> m_scene->background_color.y >> m_scene->background_color.z;
        stream.clear();
    }

    element = root->FirstChildElement("ShadowRayEpsilon");
    if (element)
    {
        stream << element->GetText() << std::endl;
        stream >> m_scene->shadow_ray_epsilon;
        stream.clear();
    }


    //Lights
    std::vector<IntensityLight> intensity_lights;
    std::vector<RectangleLight> rectangle_lights;
    glm::vec3 intensity;
    glm::vec3 edge_vector1, edge_vector2;
    float falloff_angle, cutoff_angle;

    if (element = root->FirstChildElement("Lights"))
    {

        //Point lights
        element = element->FirstChildElement("PointLight");
        while (element)
        {
            child = element->FirstChildElement("Position");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: PointLight Position is not specified.");
            }
            child = element->FirstChildElement("Intensity");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: PointLight Intensity is not specified.");
            }

            stream >> position.x >> position.y >> position.z;
            stream >> intensity.x >> intensity.y >> intensity.z;
            intensity_lights.push_back(IntensityLight(position, intensity));

            element = element->NextSiblingElement("PointLight");
            stream.clear();
        }
        //Spot lights
        element = root->FirstChildElement("Lights");
        element = element->FirstChildElement("SpotLight");
        while (element)
        {
            child = element->FirstChildElement("Position");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Position is not specified.");
            }
            child = element->FirstChildElement("Direction");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Direction is not specified.");
            }
            child = element->FirstChildElement("Intensity");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Intensity is not specified.");
            }
            child = element->FirstChildElement("FalloffAngle");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight FalloffAngle is not specified.");
            }
            child = element->FirstChildElement("CutoffAngle");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight CutoffAngle is not specified.");
            }

            stream >> position.x >> position.y >> position.z;
            stream >> gaze.x >> gaze.y >> gaze.z;
            stream >> intensity.x >> intensity.y >> intensity.z;
            stream >> falloff_angle >> cutoff_angle;
            intensity_lights.push_back(IntensityLight(position, intensity, gaze, falloff_angle, cutoff_angle));

            element = element->NextSiblingElement("SpotLight");
            stream.clear();
        }
        if (intensity_lights.size())
        {
            m_scene->addIntensityLights(intensity_lights);
        }
        //Rectangle lights
        element = root->FirstChildElement("Lights");
        element = element->FirstChildElement("RectangleLight");
        while (element)
        {
            child = element->FirstChildElement("Radiance");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Radiance is not specified.");
            }
            child = element->FirstChildElement("Position");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Position is not specified.");
            }
            child = element->FirstChildElement("Edge1");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Edge1 is not specified.");
            }
            child = element->FirstChildElement("Edge2");
            if (child)
            {
                stream << child->GetText() << std::endl;
            }
            else
            {
                throw std::runtime_error("Error: SpotLight Edge2 is not specified.");
            }

            stream >> intensity.x >> intensity.y >> intensity.z;
            stream >> position.x >> position.y >> position.z;
            stream >> edge_vector1.x >> edge_vector1.y >> edge_vector1.z;
            stream >> edge_vector2.x >> edge_vector2.y >> edge_vector2.z;
            rectangle_lights.push_back(RectangleLight(intensity, position, edge_vector1, edge_vector2));

            element = element->NextSiblingElement("RectangleLight");
            stream.clear();
        }
        if (rectangle_lights.size())
        {
            m_scene->addRectangleLights(rectangle_lights);
        }
    }


    //Textures
    std::string image_path, filter_mode_str, address_mode_str, perlin_mode_str;
    Texture::FilterMode filter_mode;
    Texture::AddressMode address_mode;
    Texture::PerlinMode perlin_mode;
    float perlin_scale;

    element = root->FirstChildElement("Textures");
    if (element)
    {
        element = element->FirstChildElement("Texture");
        while (element)
        {
            if (element->Attribute("type"))
            {
                if (std::string(element->Attribute("type")) == "Image")
                {
                    child = element->FirstChildElement("ImagePath");
                    if (child)
                    {
                        stream << child->GetText() << std::endl;
                    }
                    else
                    {
                        throw std::runtime_error("Error: Texture ImagePath is not specified.");
                    }
                    child = element->FirstChildElement("FilterMode");
                    if (child)
                    {
                        stream << child->GetText() << std::endl;
                    }
                    else
                    {
                        stream << "nearest" << std::endl;
                    }
                    child = element->FirstChildElement("AddressMode");
                    if (child)
                    {
                        stream << child->GetText() << std::endl;
                    }
                    else
                    {
                        stream << "repeat" << std::endl;
                    }

                    stream >> image_path >> filter_mode_str >> address_mode_str;

                    //FilterMode
                    if (filter_mode_str == "nearest")
                    {
                        filter_mode = Texture::NEAREST;
                    }
                    else if (filter_mode_str == "bilinear")
                    {
                        filter_mode = Texture::BILINEAR;
                    }
                    else
                    {
                        throw std::runtime_error("Error: Wrong FilterMode parameter.");
                    }

                    //AddressMode
                    if (address_mode_str == "repeat")
                    {
                        address_mode = Texture::REPEAT;
                    }
                    else if (address_mode_str == "clamp")
                    {
                        address_mode = Texture::CLAMP;
                    }
                    else
                    {
                        throw std::runtime_error("Error: Wrong AddressMode parameter.");
                    }

                    TextureManager::Manager().loadImageTexture(image_path.c_str(), Texture::SampleParams(filter_mode, address_mode));
                }
                else if (std::string(element->Attribute("type")) == "Perlin")
                {
                    child = element->FirstChildElement("PerlinMode");
                    if (child)
                    {
                        stream << child->GetText() << std::endl;
                    }
                    else
                    {
                        stream << "patch" << std::endl;
                    }
                    child = element->FirstChildElement("ScalingFactor");
                    if (child)
                    {
                        stream << child->GetText() << std::endl;
                    }
                    else
                    {
                        stream << "1" << std::endl;
                    }

                    stream >> perlin_mode_str >> perlin_scale;

                    //PerlinMode
                    if (perlin_mode_str == "patch")
                    {
                        perlin_mode = Texture::PATCH;
                    }
                    else if (perlin_mode_str == "vein")
                    {
                        perlin_mode = Texture::VEIN;
                    }
                    else
                    {
                        throw std::runtime_error("Error: Wrong PerlinMode parameter.");
                    }

                    TextureManager::Manager().loadPerlinTexture(Texture::SampleParams(perlin_mode, perlin_scale));
                }
                else
                {
                    throw std::runtime_error("Error: Wrong texture type.");
                }
            }
            else
            {
                throw std::runtime_error("Error: Texture type is not specified.");
            }

            element = element->NextSiblingElement("Texture");
            stream.clear();
        }
        if (TextureManager::Manager().get_textures().size())
        {
            m_scene->addTextures(TextureManager::Manager().get_textures());
        }
    }


    //Materials
    element = root->FirstChildElement("Materials");

    std::vector<Material> materials;
    glm::vec3 dont_care3f(0.0f, 0.0f, 0.0f);
    float dont_care1f = 0.0f;
    glm::vec3 emission, diffuse, specular, mirror, tint;
    int texture_id;
    float tint_distance, ior, scattering_coefficient, anisotropy;

    element = element->FirstChildElement("Material");
    while (element)
    {
        if (element->Attribute("type"))
        {
            if (std::string(element->Attribute("type")) == "Emissive")
            {
                child = element->FirstChildElement("Emission");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material Emission is not specified.");
                }

                stream >> emission.x >> emission.y >> emission.z;

                materials.push_back(Material(emission, dont_care3f, dont_care3f, dont_care3f, dont_care3f, dont_care1f,
                    dont_care1f, dont_care1f, nullptr, dont_care1f, Material::EMISSIVE));
            }
            else if (std::string(element->Attribute("type")) == "Lambertian")
            {
                child = element->FirstChildElement("DiffuseReflectance");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material DiffuseReflectance is not specified.");
                }
                child = element->FirstChildElement("DiffuseTexture");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    stream << "-1" << std::endl;
                }

                stream >> diffuse.x >> diffuse.y >> diffuse.z;
                stream >> texture_id;

                const Texture* diffuse_texture_ptr = nullptr;
                if (texture_id >= 0)
                {
                    diffuse_texture_ptr = m_scene->textures.first + texture_id;
                }

                materials.push_back(Material(dont_care3f, diffuse, dont_care3f, dont_care3f, dont_care3f, dont_care1f, dont_care1f,
                    dont_care1f, diffuse_texture_ptr, dont_care1f, Material::LAMBERTIAN));
            }
            else if (std::string(element->Attribute("type")) == "PerfectSpecular")
            {
                child = element->FirstChildElement("SpecularReflectance");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material SpecularReflectance is not specified.");
                }

                stream >> mirror.x >> mirror.y >> mirror.z;

                materials.push_back(Material(dont_care3f, dont_care3f, dont_care3f, mirror, dont_care3f, dont_care1f, dont_care1f,
                    dont_care1f, nullptr, dont_care1f, Material::PERFECT_SPECULAR));
            }
            else if (std::string(element->Attribute("type")) == "PerfectRefractive")
            {
                child = element->FirstChildElement("Tint");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material Tint is not specified.");
                }
                child = element->FirstChildElement("TintDistance");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material TintDistance is not specified.");
                }
                child = element->FirstChildElement("IOR");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material IOR is not specified.");
                }

                stream >> tint.x >> tint.y >> tint.z;
                stream >> tint_distance;
                stream >> ior;

                materials.push_back(Material(dont_care3f, dont_care3f, dont_care3f, dont_care3f, tint, tint_distance, dont_care1f,
                    dont_care1f, nullptr, ior, Material::PERFECT_REFRACTIVE));
            }
            else if (std::string(element->Attribute("type")) == "Translucent")
            {
                child = element->FirstChildElement("Tint");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material Tint is not specified.");
                }
                child = element->FirstChildElement("TintDistance");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material TintDistance is not specified.");
                }
                child = element->FirstChildElement("IOR");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material IOR is not specified.");
                }
                child = element->FirstChildElement("ScatteringCoefficient");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material ScatteringCoefficient is not specified.");
                }
                child = element->FirstChildElement("Anisotropy");
                if (child)
                {
                    stream << child->GetText() << std::endl;
                }
                else
                {
                    throw std::runtime_error("Error: Material Anisotropy is not specified.");
                }

                stream >> tint.x >> tint.y >> tint.z;
                stream >> tint_distance;
                stream >> ior;
                stream >> scattering_coefficient;
                stream >> anisotropy;

                materials.push_back(Material(dont_care3f, dont_care3f, dont_care3f, dont_care3f, tint, tint_distance, scattering_coefficient,
                    anisotropy, nullptr, ior, Material::TRANSLUCENT));
            }
            else
            {
                throw std::runtime_error("Error: Wrong material type.");
            }
        }
        else
        {
            throw std::runtime_error("Error: Material type is not specified.");
        }

        element = element->NextSiblingElement("Material");
        stream.clear();
    }
    if (materials.size())
    {
        m_scene->addMaterials(materials);
    }


    //Objects-Spheres
    std::vector<Sphere> spheres;
    unsigned int material_id;

    element = root->FirstChildElement("Objects");

    if (!element)
    {
        throw std::runtime_error("Error: There is no object to render.");
    }

    element = element->FirstChildElement("Sphere");
    while (element)
    {
        glm::mat4 sphere_transformation;
        child = element->FirstChildElement("Transformation");
        if (child)
        {
            glm::vec3 scaling, rotation, translation;
            float angle;
            auto child2 = child->FirstChildElement("Scaling");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "1 1 1" << std::endl;
            }
            child2 = child->FirstChildElement("Rotation");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "1 0 0 0" << std::endl;
            }
            child2 = child->FirstChildElement("Translation");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "0 0 0" << std::endl;
            }

            stream >> scaling.x >> scaling.y >> scaling.z;
            stream >> rotation.x >> rotation.y >> rotation.z >> angle;
            stream >> translation.x >> translation.y >> translation.z;
            stream.clear();

            sphere_transformation = glm::scale(scaling) * sphere_transformation;
            sphere_transformation = glm::rotate(glm::radians(angle), rotation) * sphere_transformation;
            sphere_transformation = glm::translate(translation) * sphere_transformation;
        }

        child = element->FirstChildElement("Material");
        if (child)
        {
            stream << child->GetText() << std::endl;
        }
        else
        {
            throw std::runtime_error("Error: Sphere Material is not specified.");
        }

        stream >> material_id;

        spheres.push_back(Sphere(Instance(sphere_transformation), material_id));

        element = element->NextSiblingElement("Sphere");
        stream.clear();
    }
    if (spheres.size())
    {
        m_scene->addSpheres(spheres);
    }


    //Objects-Mesh
    std::map<int, Mesh> meshes;

    element = root->FirstChildElement("Objects");
    element = element->FirstChildElement("Mesh");
    while (element)
    {
        int mesh_id;
        if (element->Attribute("id"))
        {
            mesh_id = std::stoi(element->Attribute("id"));
        }
        else
        {
            throw std::runtime_error("Error: Mesh id is not specified.");
        }

        //Transformation
        glm::mat4 mesh_transformation;
        child = element->FirstChildElement("Transformation");
        if (child)
        {
            glm::vec3 scaling, rotation, translation;
            float angle;
            auto child2 = child->FirstChildElement("Scaling");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "1 1 1" << std::endl;
            }
            child2 = child->FirstChildElement("Rotation");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "1 0 0 0" << std::endl;
            }
            child2 = child->FirstChildElement("Translation");
            if (child2)
            {
                stream << child2->GetText() << std::endl;
            }
            else
            {
                stream << "0 0 0" << std::endl;
            }

            stream >> scaling.x >> scaling.y >> scaling.z;
            stream >> rotation.x >> rotation.y >> rotation.z >> angle;
            stream >> translation.x >> translation.y >> translation.z;
            stream.clear();

            mesh_transformation = glm::scale(scaling) * mesh_transformation;
            mesh_transformation = glm::rotate(glm::radians(angle), rotation) * mesh_transformation;
            mesh_transformation = glm::translate(translation) * mesh_transformation;
        }

        //Material
        child = element->FirstChildElement("Material");
        if (child)
        {
            stream << child->GetText() << std::endl;
        }
        else
        {
            throw std::runtime_error("Error: Mesh Material is not specified.");
        }

        stream >> material_id;

        int instance_of = -1;
        auto element_instance_of = root->FirstChildElement("Objects")->FirstChildElement("Mesh");
        if (element->Attribute("instanceOf"))
        {
            instance_of = std::stoi(element->Attribute("instanceOf"));
            while (std::stoi(element_instance_of->Attribute("id")) != instance_of)
            {
                element_instance_of = element_instance_of->NextSiblingElement("Mesh");
            }
        }

        //Build the mesh.
        std::vector<Triangle> mesh_triangles;
        BBox mesh_bbox;

        if (instance_of >= 0)
        {
            child = element_instance_of->FirstChildElement("Data");
        }
        else
        {
            child = element->FirstChildElement("Data");
        }

        //Use assimp to load the model.
        if (child)
        {
            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile(child->GetText(),
                aiProcess_Triangulate |
                aiProcess_JoinIdenticalVertices |
                aiProcess_GenSmoothNormals);

            if (!scene)
            {
                throw std::runtime_error("Error: Assimp cannot load the model.");
            }

            if (scene->mNumMeshes > 1)
            {
                throw std::runtime_error("Error: More than one mesh to handle.");
            }

            aiMesh* mesh = scene->mMeshes[0];
            int face_count = mesh->mNumFaces;
            glm::vec3 face_vertices[3];
            glm::vec3 face_normals[3];
            glm::vec2 face_uvs[3];
            for (int i = 0; i < face_count; ++i)
            {
                const aiFace& face = mesh->mFaces[i];
                for (int k = 0; k < 3; ++k)
                {
                    const aiVector3D& position = mesh->mVertices[face.mIndices[k]];
                    const aiVector3D& normal = mesh->mNormals[face.mIndices[k]];
                    aiVector3D uv = mesh->HasTextureCoords(0) ? mesh->mTextureCoords[0][face.mIndices[k]] : aiVector3D(0.0f, 0.0f, 0.0f);

                    face_vertices[k] = glm::vec3(position.x, position.y, position.z);
                    face_normals[k] = glm::vec3(normal.x, normal.y, normal.z);
                    face_uvs[k] = glm::vec2(uv.x, uv.y);
                }

                mesh_bbox.extend(mesh_transformation * glm::vec4(face_vertices[0], 1.0f));
                mesh_bbox.extend(mesh_transformation * glm::vec4(face_vertices[1], 1.0f));
                mesh_bbox.extend(mesh_transformation * glm::vec4(face_vertices[2], 1.0f));

                mesh_triangles.push_back(Triangle(face_vertices[0], face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0],
                    face_normals[0], face_normals[1], face_normals[2],
                    face_uvs[0], face_uvs[1], face_uvs[2]));
            }
        }
        else
        {
            throw std::runtime_error("Error: Mesh Data is not specified.");
        }

        Mesh mesh(Instance(mesh_transformation), mesh_bbox, material_id);

        if (instance_of >= 0)
        {
            mesh.createInstanceMesh(meshes[instance_of]);
        }
        else
        {
            mesh.createBaseMesh(mesh_triangles);
        }
        meshes[mesh_id] = mesh;

        element = element->NextSiblingElement("Mesh");
        stream.clear();
    }
    if (meshes.size())
    {
        auto meshes_size = meshes.size();
        std::vector<Mesh> meshes_vector(meshes_size);

        for (int i = 0; i < meshes_size; ++i)
        {
            meshes_vector[i] = meshes[i];
        }

        m_scene->addMeshes(meshes_vector);
    }
}

//Photo mode
void World::photo()
{
    auto& image = m_renderer->render(*m_camera, *m_scene, 0);
    image.save(std::string("C://Users//Mustafa//Desktop//") + m_image_name.c_str());
}

//Video mode
void World::video()
{
    m_system.reset(new System(m_screen_width, m_screen_height));

    Timer fps_timer;
    Timer timer;
    fps_timer.start();
    timer.start();
    int counter = 0;
    int sample_count = 0;
    int jitter_dimension = 2;
    int unbiased_at = 1;
    bool save_image = false;

    while (true)
    {
        auto& image = m_renderer->render(*m_camera, *m_scene, sample_count);
        m_system->updateWindow(image.get_image());

        ++sample_count;
        ++counter;
        if (sample_count >= unbiased_at)
        {
            unbiased_at += (jitter_dimension * jitter_dimension);
            ++jitter_dimension;

            if (save_image)
            {
                image.save(std::string("C://Users//Mustafa//Desktop//") + "_" + std::to_string(sample_count) + "_" + m_image_name.c_str());
                save_image = false;
            }
        }
        m_system->setWindowTitle("FPS: " + std::to_string(counter / fps_timer.getTime()) + " -_- Sample Count: " + std::to_string(sample_count) +
            " -_- Unbiased At: " + std::to_string(unbiased_at));

        //CAMERA INPUTS//
        //
        //
        //
        auto delta_time = timer.getTime();
        timer.start();

        //Poll events before handling.
        glfwPollEvents();

        //Handle keyboard inputs.
        if (m_system->queryKey(GLFW_KEY_ESCAPE, GLFW_PRESS))
        {
            break;
        }
        if (m_system->queryKey(GLFW_KEY_ENTER, GLFW_PRESS))
        {
            save_image = true;
        }
        auto right_disp = 0.0f;
        auto forward_disp = 0.0f;
        if (m_system->queryKey(GLFW_KEY_W, GLFW_PRESS))
        {
            forward_disp += m_camera_speed * delta_time;
        }
        if (m_system->queryKey(GLFW_KEY_S, GLFW_PRESS))
        {
            forward_disp -= m_camera_speed * delta_time;
        }
        if (m_system->queryKey(GLFW_KEY_D, GLFW_PRESS))
        {
            right_disp += m_camera_speed * delta_time;
        }
        if (m_system->queryKey(GLFW_KEY_A, GLFW_PRESS))
        {
            right_disp -= m_camera_speed * delta_time;
        }

        //Handle mouse inputs.
        auto current_x = 0.0;
        auto current_y = 0.0;

        m_system->getCursorPosition(current_x, current_y);

        static auto last_x = current_x;
        static auto last_y = current_y;

        auto x_offset = last_x - current_x;
        auto y_offset = last_y - current_y;
        last_x = current_x;
        last_y = current_y;

        x_offset *= 0.002f;
        y_offset *= 0.002f;

        //Update m_camera's position and orientation.
        if (right_disp != 0 || forward_disp != 0)
        {
            m_camera->move(right_disp, forward_disp);
            fps_timer.start();
            counter = 0;
            sample_count = 0;
            jitter_dimension = 2;
            unbiased_at = 1;
        }
        if (x_offset != 0 || y_offset != 0)
        {
            m_camera->rotate(x_offset, y_offset);
            fps_timer.start();
            counter = 0;
            sample_count = 0;
            jitter_dimension = 2;
            unbiased_at = 1;
        }
        //
        //
        //
        //CAMERA INPUTS//
    }
}