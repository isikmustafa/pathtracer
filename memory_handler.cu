//raytracer.mustafaisik.net//

#include "memory_handler.cuh"
#include "cuda_utils.cuh"

#include "cuda_runtime.h"

Memory MemoryHandler::allocateOnDevice(size_t data_size, const Memory& src)
{
    void* ptr = nullptr;
    HANDLE_ERROR(cudaMalloc((void **)&ptr, data_size));
    if (!ptr)
    {
        throw std::runtime_error("Error: Device memory allocation cannot be performed");
    }
    Memory memory(Memory::DEVICE, ptr);
    m_pointers.push_back(memory);

    if (src.type == Memory::DEVICE)
    {
        HANDLE_ERROR(cudaMemcpy(ptr, src.pointer, data_size, cudaMemcpyDeviceToDevice));
    }

    else if (src.type == Memory::HOST)
    {
        HANDLE_ERROR(cudaMemcpy(ptr, src.pointer, data_size, cudaMemcpyHostToDevice));
    }

    return memory;
}

Memory MemoryHandler::allocateOnHost(size_t data_size, const Memory& src)
{
    void* ptr = nullptr;
    ptr = malloc(data_size);
    if (!ptr)
    {
        throw std::runtime_error("Error: Host memory allocation cannot be performed");
    }
    Memory memory(Memory::HOST, ptr);
    m_pointers.push_back(memory);

    if (src.type == Memory::DEVICE)
    {
        HANDLE_ERROR(cudaMemcpy(ptr, src.pointer, data_size, cudaMemcpyDeviceToHost));
    }

    else if (src.type == Memory::HOST)
    {
        HANDLE_ERROR(cudaMemcpy(ptr, src.pointer, data_size, cudaMemcpyHostToHost));
    }

    return memory;
}

//Frees the pointer contained by the memory structure only if it is allocated by the MemoryHandler.
//Returns true if it is a successful operation.
bool MemoryHandler::free(const Memory& memory)
{
    for (auto& mem : m_pointers)
    {
        if (mem == memory)
        {
            if (mem.type == Memory::DEVICE)
            {
                HANDLE_ERROR(cudaFree(mem.pointer));
            }

            else if (mem.type == Memory::HOST)
            {
                ::free(mem.pointer);
            }

            mem.pointer = nullptr;
            m_pointers.remove(mem);

            return true;
        }
    }

    return false;
}

MemoryHandler& MemoryHandler::Handler()
{
    static MemoryHandler handler;
    return handler;
}

//Returns true if it is a successful operation.
bool MemoryHandler::copy(const Memory& dst, const Memory& src, size_t data_size)
{
    if (dst.type == Memory::HOST && src.type == Memory::HOST)
    {
        HANDLE_ERROR(cudaMemcpy(dst.pointer, src.pointer, data_size, cudaMemcpyHostToHost));
    }

    else if (dst.type == Memory::DEVICE && src.type == Memory::HOST)
    {
        HANDLE_ERROR(cudaMemcpy(dst.pointer, src.pointer, data_size, cudaMemcpyHostToDevice));
    }

    else if (dst.type == Memory::HOST && src.type == Memory::DEVICE)
    {
        HANDLE_ERROR(cudaMemcpy(dst.pointer, src.pointer, data_size, cudaMemcpyDeviceToHost));
    }

    else if (dst.type == Memory::DEVICE && src.type == Memory::DEVICE)
    {
        HANDLE_ERROR(cudaMemcpy(dst.pointer, src.pointer, data_size, cudaMemcpyDeviceToDevice));
    }

    else
    {
        return false;
    }

    return true;
}

MemoryHandler::~MemoryHandler()
{
    for (auto& memory : m_pointers)
    {
        if (memory.type == Memory::DEVICE)
        {
            HANDLE_ERROR(cudaFree(memory.pointer));
        }

        else if (memory.type == Memory::HOST)
        {
            ::free(memory.pointer);
        }

        memory.pointer = nullptr;
    }
}