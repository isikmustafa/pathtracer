//raytracer.mustafaisik.net//

#pragma once

#include "texture.cuh"

#include <map>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "cuda_runtime.h"

//Singleton class which manages texture loading.
class TextureManager
{
public:
    TextureManager(const TextureManager&) = delete;
    TextureManager(TextureManager&&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;
    TextureManager& operator=(TextureManager&&) = delete;

    static TextureManager& Manager();

    void loadPerlinTexture(const Texture::SampleParams& sample_params);
    void loadImageTexture(const std::string& filepath, const Texture::SampleParams& sample_params);
    
    //Getters
    const std::vector<Texture>& get_textures() const
    {
        return m_textures;
    }

private:
    std::vector<Texture> m_textures;
    std::map<std::string, cudaArray*> m_namearray_pair;

private:
    TextureManager() = default;
    ~TextureManager();
};