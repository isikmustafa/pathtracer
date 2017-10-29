//raytracer.mustafaisik.net//

#pragma once

#include "gl/glsl_program.h"
#include "gl/quad.h"

#include <string>

struct GLFWwindow;
struct cudaGraphicsResource;

class System
{
public:
    System(int screen_width, int screen_height);
    ~System();
    System(const System&) = delete;
    System(System&&) = delete;
    System& operator=(const System&) = delete;
    System& operator=(System&&) = delete;

    void updateWindow(unsigned char* image);
    bool queryKey(int key, int condition) const;
    void setWindowTitle(const std::string& title) const;
    void getCursorPosition(double& x, double& y) const;

private:
    //CUDA-GL Interop
    GLSLProgram m_window_shader;
    Quad m_window_quad;
    cudaGraphicsResource* m_resource;
    GLuint m_cuda_output_texture;

    //Regular window variables
    GLFWwindow* m_window;
    int m_screen_width, m_screen_height;
};