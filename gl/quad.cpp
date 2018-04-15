#include "quad.h"
#include "glsl_program.h"

#include <assert.h>

Quad::Quad()
    : m_vertex_array(0)
    , m_vertex_buffer(0)
{}

Quad::~Quad()
{
    if (m_vertex_buffer)
    {
        glDeleteBuffers(1, &m_vertex_buffer);
        m_vertex_buffer = 0;
    }

    if (m_vertex_array)
    {
        glDeleteVertexArrays(1, &m_vertex_array);
        m_vertex_array = 0;
    }
}

void Quad::create()
{
    GLfloat vertices[] =
    {
        //NDC coordinates for the quad.
        //Positions     //Texture coordinates
        -1.0f,  1.0f,   0.0f, 1.0f,
        -1.0f, -1.0f,   0.0f, 0.0f,
        1.0f , -1.0f,   1.0f, 0.0f,

        -1.0f,  1.0f,   0.0f, 1.0f,
        1.0f , -1.0f,   1.0f, 0.0f,
        1.0f ,  1.0f,   1.0f, 1.0f
    };

    glGenVertexArrays(1, &m_vertex_array);
    glGenBuffers(1, &m_vertex_buffer);

    assert(m_vertex_array);
    assert(m_vertex_buffer);

    glBindVertexArray(m_vertex_array);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(GLfloat)));
    glBindVertexArray(0);
}

void Quad::draw(const GLSLProgram& program) const
{
    program.use();

    glBindVertexArray(m_vertex_array);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}