//raytracer.mustafaisik.net//

#include "texture_manager.cuh"
#include "cuda_utils.cuh"

#include <FreeImage/FreeImage.h>

TextureManager& TextureManager::Manager()
{
    static TextureManager manager;
    return manager;
}

void TextureManager::loadPerlinTexture(const Texture::SampleParams& sample_params)
{
    m_textures.push_back(Texture(sample_params, Texture::PERLIN, 0));
}

//Checks if the "filepath" image exists or not.
//If it is loaded before, it uses the existing data for it.
//If it is not loaded before, it creates the new data.
void TextureManager::loadImageTexture(const std::string& filepath, const Texture::SampleParams& sample_params)
{
    auto pair = m_namearray_pair.find(filepath.c_str());

    cudaArray* cuda_array = nullptr;
    //If it is loaded before.
    if (pair != m_namearray_pair.end())
    {
        cuda_array = pair->second;
    }
    //If it is not loaded before.
    else
    {
        FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
        FIBITMAP* dib_raw = nullptr;
        BYTE* bits = nullptr;

        fif = FreeImage_GetFileType(filepath.c_str(), 0);
        if (fif == FIF_UNKNOWN)
        {
            fif = FreeImage_GetFIFFromFilename(filepath.c_str());
        }
        if (fif == FIF_UNKNOWN)
        {
            throw std::runtime_error("Error: Unknown image file format");
        }
        if (FreeImage_FIFSupportsReading(fif))
        {
            dib_raw = FreeImage_Load(fif, filepath.c_str());
        }
        if (!dib_raw)
        {
            throw std::runtime_error("Error: Failed to load the image file");
        }

        auto dib = FreeImage_ConvertTo32Bits(dib_raw);
        FreeImage_FlipVertical(dib);

        bits = FreeImage_GetBits(dib);
        auto image_width = FreeImage_GetWidth(dib);
        auto image_height = FreeImage_GetHeight(dib);

        //Create the texture on device.
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); //RGBA

        HANDLE_ERROR(cudaMallocArray(&cuda_array, &channel_desc, image_width, image_height));
        HANDLE_ERROR(cudaMemcpyToArray(cuda_array, 0, 0, bits, image_width * image_height * 4, cudaMemcpyHostToDevice));

        m_namearray_pair.insert(std::make_pair(filepath.c_str(), cuda_array));
        FreeImage_Unload(dib_raw);
        FreeImage_Unload(dib);
    }

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaTextureAddressMode(sample_params.address_mode);
    tex_desc.addressMode[1] = cudaTextureAddressMode(sample_params.address_mode);
    tex_desc.filterMode = cudaTextureFilterMode(sample_params.filter_mode);
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    cudaTextureObject_t texture = 0;
    HANDLE_ERROR(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));

    m_textures.push_back(Texture(sample_params, Texture::IMAGE, texture));
}

TextureManager::~TextureManager()
{
    for (auto& texture : m_textures)
    {
        if (texture.get_texture_type() == Texture::IMAGE)
        {
            cudaDestroyTextureObject(texture.get_texture());
        }
    }

    for (auto& pair : m_namearray_pair)
    {
        cudaFreeArray(pair.second);
    }
}