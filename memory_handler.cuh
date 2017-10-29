//raytracer.mustafaisik.net//

#pragma once

#include <list>

struct Memory
{
public:
    enum Type
    {
        HOST,
        DEVICE,
        UNDEFINED
    };

public:
    Type type;
    void* pointer;

public:
    Memory()
        : type(UNDEFINED)
        , pointer(nullptr)
    {}

    Memory(Type p_type, void* p_pointer)
        : type(p_type)
        , pointer(p_pointer)
    {}

    bool operator == (const Memory& memory)
    {
        return type == memory.type && pointer == memory.pointer;
    }
};

//Singleton class which manages memory allocations and deallocations.
class MemoryHandler
{
public:
    MemoryHandler(const MemoryHandler&) = delete;
    MemoryHandler(MemoryHandler&&) = delete;
    MemoryHandler& operator=(const MemoryHandler&) = delete;
    MemoryHandler& operator=(MemoryHandler&&) = delete;

    Memory allocateOnDevice(size_t data_size, const Memory& src = Memory());
    Memory allocateOnHost(size_t data_size, const Memory& src = Memory());
    bool free(const Memory& memory);

    static MemoryHandler& Handler();
    static bool copy(const Memory& dst, const Memory& src, size_t data_size);

private:
    std::list<Memory> m_pointers;

private:
    MemoryHandler() = default;
    ~MemoryHandler();
};