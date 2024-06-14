#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>
#include "glm/common.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/gtc/type_ptr.hpp"

class WorldTrans {
 public:
    WorldTrans() {}

    void SetScale(float scale);
    void SetRotation(float x, float y, float z);
    void SetPosition(float x, float y, float z);

    void Rotate(float x, float y, float z);

    glm::mat4 GetMatrix();

 private:
    float    m_scale    = 1.0f;
    glm::vec3 m_rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 m_pos      = glm::vec3(0.0f, 0.0f, 0.0f);
};


class ProjectionTrans {
    public:
        float FOV = 45.0f;
        float zNear = 1.0f;
        float zFar = 10.0f;
        int width, height;
        ProjectionTrans(int width, int height);
        glm::mat4 getProjectionMatrix();
};

class Camera {
public:

    Camera();

    void SetPosition(float x, float y, float z);

    void OnKeyboard(int key);

    glm::mat4 GetMatrix();

private:

    glm::vec3 m_pos;
    glm::vec3 m_target;
    glm::vec3 m_up;
    float m_speed = 1.0f;
};

#endif // _CAMERA_H_