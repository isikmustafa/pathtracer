//raytracer.mustafaisik.net//

#pragma once

#include "image.cuh"

#include "cuda_runtime.h"

class Camera;
class Scene;

class Renderer
{
public:
    Renderer(int image_width, int image_height);

    const Image& render(const Camera& camera, const Scene& scene, int sample_count) const;

private:
    Image m_image;
    int m_image_width, m_image_height;
};