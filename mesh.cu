//raytracer.mustafaisik.net//

#include "mesh.cuh"
#include "memory_handler.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ Mesh::Mesh()
    : m_instance()
    , m_bbox()
    , m_triangles(nullptr, 0)
    , m_nodes(nullptr, 0)
    , m_material_id(0)
{}

__host__ Mesh::Mesh(const Instance& instance, const BBox& bbox, unsigned int material_id)
    : m_instance(instance)
    , m_bbox(bbox)
    , m_triangles(nullptr, 0)
    , m_nodes(nullptr, 0)
    , m_material_id(material_id)
{}

//This function should be called only when the mesh is wanted be allocated on device memory.
//If an instance of a mesh is being created and this function is called then
//same set of triangles are allocated again on the device memory, which is unnecessary.
__host__ void Mesh::createBaseMesh(const std::vector<Triangle>& mesh_triangles)
{
    std::vector<Triangle> mesh_triangles_copy;

    if (!m_nodes.first)
    {
        std::vector<BVH::BVHNode> nodes;
        mesh_triangles_copy = mesh_triangles;
        BVH::buildMeshBvh(mesh_triangles_copy, nodes);

        auto size = nodes.size();
        if (size)
        {
            m_nodes.second = size;
            size_t data_size = m_nodes.second * sizeof(BVH::BVHNode);
            auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, nodes.begin()._Ptr));
            m_nodes.first = static_cast<BVH::BVHNode*>(memory.pointer);
        }
    }

    auto size = mesh_triangles_copy.size();
    if (!m_triangles.first && size)
    {
        m_triangles.second = size;
        size_t data_size = m_triangles.second * sizeof(Triangle);
        auto memory = MemoryHandler::Handler().allocateOnDevice(data_size, Memory(Memory::HOST, mesh_triangles_copy.begin()._Ptr));
        m_triangles.first = static_cast<Triangle*>(memory.pointer);
    }
}

__host__ void Mesh::createInstanceMesh(const Mesh& base_mesh)
{
    m_triangles = base_mesh.m_triangles;
    m_nodes = base_mesh.m_nodes;
}