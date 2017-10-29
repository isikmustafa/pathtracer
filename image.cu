//raytracer.mustafaisik.net//

#include "image.cuh"
#include "memory_handler.cuh"

#include <FreeImage/FreeImage.h>

__host__ Image::Image(int image_width, int image_height)
    : m_image(nullptr)
    , m_width(image_width)
    , m_height(image_height)
{
    size_t data_size = image_width * image_height * 4 * sizeof(unsigned char);

    auto memory = MemoryHandler::Handler().allocateOnDevice(data_size);
    m_image = static_cast<unsigned char*>(memory.pointer);
}

__host__ Image::~Image()
{}

__host__ void Image::save(const std::string& filepath) const
{
    size_t data_size = m_width * m_height * 4 * sizeof(unsigned char);
    auto memory = MemoryHandler::Handler().allocateOnHost(data_size, Memory(Memory::DEVICE, m_image));

    FIBITMAP* image = FreeImage_ConvertFromRawBits(static_cast<BYTE*>(memory.pointer), m_width, m_height, 4 * sizeof(unsigned char) * m_width, 32,
        FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

    FreeImage_Save(FIF_PNG, image, filepath.c_str(), 0);
    FreeImage_Unload(image);
}

__device__ void Image::setPixel(int pixel_index, const glm::vec3& color)
{
    pixel_index *= 4;
    m_image[pixel_index] = static_cast<unsigned char>(color.z);
    m_image[pixel_index + 1] = static_cast<unsigned char>(color.y);
    m_image[pixel_index + 2] = static_cast<unsigned char>(color.x);
    m_image[pixel_index + 3] = 255;
}