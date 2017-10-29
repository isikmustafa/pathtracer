//raytracer.mustafaisik.net//

#pragma once

#include <iostream>
#include <memory>
#include <string>

class System;
class Renderer;
class Camera;
class Scene;

class World
{
public:
    World();
    ~World();

    void loadScene(const std::string& filepath);
    void photo();
    void video();

private:
    std::unique_ptr<System> m_system;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<Scene> m_scene;
    std::string m_image_name;
    int m_screen_width, m_screen_height;
    double m_camera_speed;
};