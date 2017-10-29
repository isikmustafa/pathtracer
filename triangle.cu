//raytracer.mustafaisik.net//

#include "triangle.cuh"

__host__ Triangle::Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2,
    const glm::vec2& texcoord0, const glm::vec2& texcoord1, const glm::vec2& texcoord2)
    : m_v0(v0)
    , m_edge1(edge1)
    , m_edge2(edge2)
    , m_normal0(glm::normalize(glm::cross(edge1, edge2)))
    , m_normal1(m_normal0)
    , m_normal2(m_normal0)
    , m_texcoord0(texcoord0)
    , m_texcoord1(texcoord1)
    , m_texcoord2(texcoord2)
{}

__host__ Triangle::Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2, const glm::vec3& normal0, const glm::vec3& normal1, const glm::vec3& normal2,
    const glm::vec2& texcoord0, const glm::vec2& texcoord1, const glm::vec2& texcoord2)
    : m_v0(v0)
    , m_edge1(edge1)
    , m_edge2(edge2)
    , m_normal0(glm::normalize(normal0))
    , m_normal1(glm::normalize(normal1))
    , m_normal2(glm::normalize(normal2))
    , m_texcoord0(texcoord0)
    , m_texcoord1(texcoord1)
    , m_texcoord2(texcoord2)
{}

__host__ BBox Triangle::getBBox() const
{
    auto v1 = m_edge1 + m_v0;
    auto v2 = m_edge2 + m_v0;

    return BBox(glm::min(glm::min(v1, v2), m_v0), glm::max(glm::max(v1, v2), m_v0));
}