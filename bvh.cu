//raytracer.mustafaisik.net//

#include "bvh.cuh"
#include "instance.cuh"
#include "math_utils.cuh"
#include "mesh.cuh"
#include "memory_handler.cuh"
#include "timer.cuh"
#include "triangle.cuh"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stack>
#include <queue>

//Contructs a BVH tree with SAH.
//Reorders the "mesh_triangles".
//Creates the context of "nodes".
//Children and triangle information contained by inner nodes and leaves are
//represented by integer indices. Therefore, the order of elements in "mesh_triangles"
//and "nodes" should not be changed.
void BVH::buildMeshBvh(std::vector<Triangle>& mesh_triangles, std::vector<BVHNode>& nodes)
{
    struct Bin
    {
        BBox bbox;
        std::vector<int> triangle_id_list;
    };

    struct SNode
    {
        Bin nbin;
        int parent_id;
        int left_or_right;

        SNode(const Bin& p_nbin, int p_parent_id, int p_left_or_right)
            : nbin(p_nbin)
            , parent_id(p_parent_id)
            , left_or_right(p_left_or_right)
        {}
    };

    Timer build_timer;
    build_timer.start();
    int triangle_count = 0;

    //BBoxes&centroids
    auto size = mesh_triangles.size();
    std::vector<BBox> bboxes(size);
    std::vector<glm::vec3> centroids(size);

    Bin root_bin;
    root_bin.triangle_id_list.resize(size);
    std::iota(std::begin(root_bin.triangle_id_list), std::end(root_bin.triangle_id_list), 0);

    for (int i = 0; i < size; ++i)
    {
        bboxes[i] = mesh_triangles[i].getBBox();
        centroids[i] = (bboxes[i].max + bboxes[i].min) / 2.0f;
        root_bin.bbox.extend(bboxes[i]);
    }

    //Temporary list
    std::vector<Triangle> temp_triangle_list(mesh_triangles);
    mesh_triangles.clear();
    mesh_triangles.reserve(size);

    //Fixed leaf and bin sizes
    const int cMaxLeafSize = fmaxf(log2f(static_cast<float>(size)) / 2, 4);
    const int cBinSize = 16;

    //Initialize stack
    std::stack<SNode> stack;
    stack.push(SNode(root_bin, -1, -1));

    while (!stack.empty())
    {
        auto& node = stack.top();
        auto& bbox = node.nbin.bbox;
        auto& id_list = node.nbin.triangle_id_list;
        bool make_node = false;

        //Inform parents about id of the child.
        if (node.parent_id >= 0)
        {
            //left
            if (node.left_or_right == 0)
            {
                nodes[node.parent_id].left_node = nodes.size();
            }
            //right
            else
            {
                nodes[node.parent_id].right_node = nodes.size();
            }
        }

        auto list_size = id_list.size();
        if (list_size <= cMaxLeafSize)
        {
            make_node = true;
        }

        else
        {
            auto edges = bbox.max - bbox.min;

            int axis = 0;
            if (edges.y > edges.x)
            {
                axis = 1;
            }
            if (edges.z > edges.y && edges.z > edges.x)
            {
                axis = 2;
            }

            //Fill the bins.
            Bin bins[cBinSize];
            auto constant_term = cBinSize * (1 - math::float_epsilon) / (bbox.max[axis] - bbox.min[axis]);
            for (auto i : id_list)
            {
                auto bin_index = static_cast<int>((centroids[i][axis] - bbox.min[axis]) * constant_term);
                bins[bin_index].bbox.extend(bboxes[i]);
                bins[bin_index].triangle_id_list.push_back(i);
            }

            //Find minimum cost
            auto minimum_cost = math::float_max;
            int minimum_cost_bin_index = -1;
            for (int i = 0; i < cBinSize - 1; ++i)
            {
                //left
                BBox left_part;
                int left_object_count = 0;
                int j = 0;
                for (; j <= i; ++j)
                {
                    left_part.extend(bins[j].bbox);
                    left_object_count += bins[j].triangle_id_list.size();
                }
                //right
                BBox right_part;
                int right_object_count = 0;
                for (; j < cBinSize; ++j)
                {
                    right_part.extend(bins[j].bbox);
                    right_object_count += bins[j].triangle_id_list.size();
                }

                //If it is a better cost and presents two unempty halves.
                auto current_cost = left_part.getSurfaceArea() * left_object_count + right_part.getSurfaceArea() * right_object_count;
                if (current_cost < minimum_cost && left_object_count && right_object_count)
                {
                    minimum_cost = current_cost;
                    minimum_cost_bin_index = i;
                }
            }

            //If all the triangles are piled up in just one bin,
            //make it a node. Do not split it again.
            if (minimum_cost == math::float_max)
            {
                make_node = true;
            }

            //Otherwise, make it an inner node.
            else
            {
                Bin left_bin, right_bin;
                for (int i = 0; i < cBinSize; ++i)
                {
                    //left
                    if (i <= minimum_cost_bin_index)
                    {
                        left_bin.bbox.extend(bins[i].bbox);
                        left_bin.triangle_id_list.insert(left_bin.triangle_id_list.end(), bins[i].triangle_id_list.begin(), bins[i].triangle_id_list.end());
                    }
                    //right
                    else
                    {
                        right_bin.bbox.extend(bins[i].bbox);
                        right_bin.triangle_id_list.insert(right_bin.triangle_id_list.end(), bins[i].triangle_id_list.begin(), bins[i].triangle_id_list.end());
                    }
                }

                stack.pop();
                stack.push(SNode(right_bin, nodes.size(), 1)); //right
                stack.push(SNode(left_bin, nodes.size(), 0)); //left
                nodes.push_back(BVHNode(left_bin.bbox, right_bin.bbox, 0, 0, 0, 0));
            }
        }

        if (make_node)
        {
            auto leaf_node_start = mesh_triangles.size();
            for (auto id : id_list)
            {
                mesh_triangles.push_back(temp_triangle_list[id]);
            }
            auto leaf_node_end = mesh_triangles.size() - 1;

            nodes.push_back(BVHNode(bbox, bbox, 0, 0, leaf_node_start, leaf_node_end));
            triangle_count += list_size;
            stack.pop();
        }
    }

    std::cout << "BVH-Sah build time: " << build_timer.getTime() << " seconds" << std::endl;
    std::cout << "Triangle count: " << triangle_count << std::endl << std::endl;
}

//Contructs a BVH tree with median splitting.
//Does not reorder "meshes".
//Creates the context of "nodes".
//Returns the bounding box of the scene.
BBox BVH::buildSceneBvh(const std::vector<Mesh>& meshes, std::vector<BVHNode>& nodes)
{
    struct QNode
    {
        BBox bbox;
        int start_index;
        int end_index;

        QNode(const BBox& p_bbox, int p_start_index, int p_end_index)
            : bbox(p_bbox)
            , start_index(p_start_index)
            , end_index(p_end_index)
        {}
    };

    //BBoxes&centroids
    auto size = meshes.size();
    std::vector<BBox> bboxes(size);
    std::vector<glm::vec3> centroids(size);
    BBox scene_bbox;

    for (int i = 0; i < size; ++i)
    {
        bboxes[i] = meshes[i].get_bbox();
        centroids[i] = (bboxes[i].max + bboxes[i].min) / 2.0f;
        scene_bbox.extend(bboxes[i]);
    }

    std::vector<int> id_list(size);
    std::iota(std::begin(id_list), std::end(id_list), 0);
    int node_id = 0;

    //Initialize queue
    std::queue<QNode> queue;
    queue.push(QNode(scene_bbox, 0, size - 1));

    while (!queue.empty())
    {
        auto& node = queue.front();
        auto& bbox = node.bbox;

        auto leaf_mesh_count = node.end_index - node.start_index + 1;
        if (leaf_mesh_count <= 1)
        {
            nodes.push_back(BVHNode(bbox, bbox, 0, 0, id_list[node.start_index], 0));
        }

        else
        {
            auto edges = bbox.max - bbox.min;

            int axis = 0;
            if (edges.y > edges.x)
            {
                axis = 1;
            }
            if (edges.z > edges.y && edges.z > edges.x)
            {
                axis = 2;
            }

            auto itr_start = id_list.begin() + node.start_index;
            auto itr_nth = id_list.begin() + (node.start_index + node.end_index) / 2;
            auto itr_end = id_list.begin() + node.end_index + 1;

            std::nth_element(itr_start, itr_nth, itr_end, [&centroids, &axis](int a, int b)
            {
                return centroids[a][axis] < centroids[b][axis];
            });

            auto end_left = node.start_index + (itr_nth - itr_start);
            BBox left_bbox, right_bbox;
            int i = node.start_index;
            for (; i <= end_left; ++i)
            {
                left_bbox.extend(bboxes[id_list[i]]);
            }
            for (; i <= node.end_index; ++i)
            {
                right_bbox.extend(bboxes[id_list[i]]);
            }

            queue.push(QNode(left_bbox, node.start_index, end_left));
            auto left_id = ++node_id;
            queue.push(QNode(right_bbox, end_left + 1, node.end_index));
            auto right_id = ++node_id;

            nodes.push_back(BVHNode(left_bbox, right_bbox, left_id, right_id, 0, 0));
        }
        queue.pop();
    }

    return scene_bbox;
}