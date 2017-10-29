//raytracer.mustafaisik.net//

#include "camera.cuh"

#include <glm/gtc/quaternion.hpp>

//screen_coordinates includes left,right,top and bottom components, respectively.
__host__ Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const glm::vec3& world_up,
    const glm::vec4& screen_coordinates, const glm::ivec2& screen_resolution,
    float aperture_radius, float focus_distance)
    : m_position(position)
    , m_up(glm::normalize(world_up))
    , m_forward(glm::normalize(-direction))
    , m_world_up(glm::normalize(world_up))
    , m_coefficients((screen_coordinates.y - screen_coordinates.x) / screen_resolution.x, (screen_coordinates.w - screen_coordinates.z) / screen_resolution.y)
    , m_base_pixels(m_coefficients.x * 0.5f + screen_coordinates.x, m_coefficients.y * 0.5f + screen_coordinates.z)
    , m_near_distance(glm::length(direction))
    , m_aperture_radius(aperture_radius)
    , m_focus_distance(focus_distance)
{
    buildView();
}

//right_disp is the displacement along the x-axis of camera space(along right vector).
//forward_disp is the displacement along the (-z)-axis of camera space(along opposite of forward vector).
__host__ void Camera::move(float right_disp, float forward_disp)
{
    m_position += m_right * right_disp;
    m_position -= m_forward * forward_disp;
}

//Camera will be rotated about up axis of the world by radian_world_up angle.
//Camera will be rotated about it's right axis by radian_right angle.
__host__ void Camera::rotate(float radian_world_up, float radian_right)
{
    auto rot_on_y = glm::angleAxis(radian_world_up, m_world_up);
    auto rot_on_right = glm::angleAxis(radian_right, m_right);

    m_right = rot_on_y * m_right;
    m_up = rot_on_y * rot_on_right * m_up;
    m_forward = rot_on_y * rot_on_right * m_forward;

    buildView();
}

__host__ void Camera::buildView()
{
    //Make sure the camera has a coordinate system whose bases are orthonormal to each other.
    m_forward = glm::normalize(m_forward);
    m_right = glm::normalize(glm::cross(m_up, m_forward));
    m_up = glm::normalize(glm::cross(m_forward, m_right));
}