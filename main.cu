//raytracer.mustafaisik.net//

#include "world.cuh"

#include "cuda_runtime.h"

int main()
{
    {
        World world;

        world.loadScene("input//cornellbox//scene-realtime.xml");
        world.video();
    }

    return 0;
}