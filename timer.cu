//raytracer.mustafaisik.net//

#include "timer.cuh"

Timer::Timer()
    : m_start_time(std::chrono::system_clock::now())
{}

//start
void Timer::start()
{
    m_start_time = std::chrono::system_clock::now();
}

//getTime
double Timer::getTime()
{
    using second = std::chrono::duration<double, std::ratio <1>>;
    return std::chrono::duration_cast<second>(std::chrono::system_clock::now() - m_start_time).count();
}