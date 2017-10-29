//raytracer.mustafaisik.net//

#include "system.cuh"
#include "cuda_utils.cuh"

#include <assert.h>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <FreeImage/FreeImage.h>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

System::System(int screen_width, int screen_height)
    : m_resource(nullptr)
    , m_cuda_output_texture(0)
    , m_window(nullptr)
    , m_screen_width(screen_width)
    , m_screen_height(screen_height)
{
    //Initialize GLFW
    auto ret = glfwInit();
    assert(ret);

    //OpenGL 4.5
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    m_window = glfwCreateWindow(m_screen_width, m_screen_height, "Morty", nullptr, nullptr);
    assert(m_window);
    glfwMakeContextCurrent(m_window);

    //Cursor settings.
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(m_window, m_screen_width / 2.0, m_screen_height / 2.0);

    //Initialize GLAD
    ret = gladLoadGL();
    assert(ret);

    //OpenGL settings
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, m_screen_width, m_screen_height);

    //Compile the glsl program
    m_window_shader.attachShader(GL_VERTEX_SHADER, "gl/window.vert");
    m_window_shader.attachShader(GL_FRAGMENT_SHADER, "gl/window.frag");
    m_window_shader.link();

    //Create the quad into which we render the final image texture
    m_window_quad.create();

    //CUDA-GL Interop Initialization
    glGenTextures(1, &m_cuda_output_texture);
    assert(m_cuda_output_texture);
    glBindTexture(GL_TEXTURE_2D, m_cuda_output_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_screen_width, m_screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    HANDLE_ERROR(cudaGraphicsGLRegisterImage(&m_resource, m_cuda_output_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
    glBindTexture(GL_TEXTURE_2D, 0);
}

System::~System()
{
    glfwTerminate();
}

void System::updateWindow(unsigned char* image)
{
    //Input
    glfwPollEvents();

    //Graphics
    //CUDA//
    static int image_data_size = m_screen_width * m_screen_height * 4 * sizeof(unsigned char);
    cudaArray* texture_ptr = nullptr;
    HANDLE_ERROR(cudaGraphicsMapResources(1, &m_resource, 0));
    HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_resource, 0, 0));
    HANDLE_ERROR(cudaMemcpyToArray(texture_ptr, 0, 0, image, image_data_size, cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &m_resource, 0));

    //GL//
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_cuda_output_texture);
    m_window_shader.setUniformIVar("final_image", { 0 });
    m_window_quad.draw(m_window_shader);
    glBindTexture(GL_TEXTURE_2D, 0);

    glfwSwapBuffers(m_window);
}

bool System::queryKey(int key, int condition) const
{
    return glfwGetKey(m_window, key) == condition;
}

void System::setWindowTitle(const std::string& title) const
{
    glfwSetWindowTitle(m_window, title.c_str());
}

void System::getCursorPosition(double& x, double& y) const
{
    glfwGetCursorPos(m_window, &x, &y);
}