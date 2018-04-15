#pragma once

#include <glad/glad.h>

class GLSLProgram;

class Quad
{
public:
    Quad();
    ~Quad();

    void create();
    void draw(const GLSLProgram& program) const;

private:
    GLuint m_vertex_array;
    GLuint m_vertex_buffer;
};