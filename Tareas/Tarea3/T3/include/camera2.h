#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>
#include "glm/common.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include <glm/gtc/quaternion.hpp>
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
        float zNear = 0.1f;
        float zFar = 100.0f;
        int width, height;
        ProjectionTrans(int width, int height);
        glm::mat4 getProjectionMatrix();
};

class Camera {
public:

    Camera();
    Camera(int WindowWidth, int WindowHeight);

    Camera(int WindowWidth, int WindowHeight, const glm::vec3& Pos, const glm::vec3& Front, const glm::vec3& Up);

    void SetPosition(float x, float y, float z);
    void SetOrientation(float AngleH, float AngleV);
    // void updateCameraVectors();
    void OnKeyboard(int key, float dt);
    void OnMouse(int x, int y);

    void OnRender(float dt);

    glm::mat4 GetMatrix();
    glm::vec3 m_pos;
    glm::vec3 m_front;

private:

    void Init();
    void Update();

    glm::vec3 m_up;
    glm::vec3 m_right;
    float m_speed = 0.1f;

    int m_windowWidth;
    int m_windowHeight;

    float m_AngleH;
    float m_AngleV;
    float dt = 0.01f;

    bool m_OnUpperEdge;
    bool m_OnLowerEdge;
    bool m_OnLeftEdge;
    bool m_OnRightEdge;

    glm::ivec2 m_mousePos;
};

#endif // _CAMERA_H_