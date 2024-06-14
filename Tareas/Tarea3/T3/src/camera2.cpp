#include "camera2.h"



void WorldTrans::SetScale(float scale) {
    m_scale = scale;
}


void WorldTrans::SetRotation(float x, float y, float z) {
    m_rotation.x = x;
    m_rotation.y = y;
    m_rotation.z = z;
}


void WorldTrans::SetPosition(float x, float y, float z) {
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
}


void WorldTrans::Rotate(float x, float y, float z) {
    m_rotation.x += x;
    m_rotation.y += y;
    m_rotation.z += z;
}


glm::mat4 WorldTrans::GetMatrix() {
    glm::mat4 Scale(glm::vec4(m_scale, 0.0f, 0.0f, 0.0f),
                    glm::vec4(0.0f, m_scale, 0.0f, 0.0f),
                    glm::vec4(0.0f, 0.0f, m_scale, 0.0f),
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    
    glm::mat4 Rotation, Rx, Ry, Rz;
    float rx = glm::radians(m_rotation[0]), ry = glm::radians(m_rotation[1]), rz = glm::radians(m_rotation[2]);
    Rx = glm::mat4(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                    glm::vec4(0.0f, cos(rx), -sin(rx), 0.0f),
                    glm::vec4(0.0f, sin(rx), cos(rx), 0.0f),
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    Ry = glm::mat4(glm::vec4(cos(ry), 0.0f, sin(ry), 0.0f),
                    glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                    glm::vec4(-sin(ry), 0.0f, cos(ry), 0.0f),
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    Rz = glm::mat4(glm::vec4(cos(rz), -sin(rz), 0.0f, 0.0f),
                    glm::vec4(sin(rz), cos(rz), 0.0f, 0.0f),
                    glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    Rotation = Rx*Ry*Rz;

    glm::mat4 Translation;
    Translation = glm::mat4(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                            glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                            glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                            glm::vec4(m_pos[0], m_pos[1], m_pos[2], 1.0f));

    glm::mat4 WorldTransformation = Translation * Rotation * Scale;

    return WorldTransformation;
}


ProjectionTrans::ProjectionTrans(int width, int height) {
    this->width = width;
    this->height = height;
}

glm::mat4 ProjectionTrans::getProjectionMatrix() {
    glm::mat4 m;
    float ar         = height / width;
    float zRange     = zNear - zFar;
    float tanHalfFOV = tan(glm::radians(FOV / 2.0f));
    m = glm::mat4(glm::vec4(1.0f/tanHalfFOV, 0.0f, 0.0f, 0.0f),
                            glm::vec4(0.0f, 1.0f/(ar*tanHalfFOV), 0.0f, 0.0f),
                            glm::vec4(0.0f, 0.0f, (-zNear - zFar)/zRange, 1.0f),
                            glm::vec4(0.0f, 0.0f, 2.0f*zFar*zNear/zRange, 1.0f));

    return m;
    // glm::mat4 Projection = glm::perspectiveFovLH(glm::radians(FOV), width, height, zNear, zFar);
}



Camera::Camera() {
    m_pos          = glm::vec3(50.0f, 50.0f, 2.0f);
    m_target       = glm::vec3(0.0f, 0.0f, -1.0f);
    m_up           = glm::vec3(0.0f, 1.0f, 0.0f);
}


void Camera::SetPosition(float x, float y, float z) {
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
}


void Camera::OnKeyboard(int key) {
    switch (key) {
    // up
    case 0:
        m_pos += (m_target * m_speed);
        break;
    // down
    case 1:
        m_pos -= (m_target * m_speed);
        break;
    // left
    case 2:
        {
            glm::vec3 Left = glm::cross(m_target,m_up);
            Left = glm::normalize(Left);
            Left *= m_speed;
            m_pos += Left;
        }
        break;
    // right
    case 3:
        {
            glm::vec3 Right = glm::cross(m_up, m_target);
            Right = glm::normalize(Right);
            Right *= m_speed;
            m_pos += Right;
        }
        break;
    // w
    case 4:
        m_pos.y += m_speed;
        break;
    // s
    case 5:
        m_pos.y -= m_speed;
        break;
    // d
    case 6:
        m_speed += 0.1f;
        // printf("Speed changed to %f\n", m_speed);
        break;
    // a
    case 7:
        m_speed -= 0.1f;
        if (m_speed < 0.1f) {
            m_speed = 0.1f;
        }
        // printf("Speed changed to %f\n", m_speed);
        break;
    }
}


glm::mat4 Camera::GetMatrix() {
    glm::mat4 CameraTranslation, CameraRotateTrans;
    CameraTranslation = glm::mat4(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
                            glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                            glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
                            glm::vec4(-m_pos[0], -m_pos[1], -m_pos[2], 1.0f));

    glm::vec3 N = m_target;
    N = glm::normalize(N);
    glm::vec3 upNorm = m_up;
    upNorm = glm::normalize(upNorm);

    glm::vec3 U = glm::normalize(glm::cross(upNorm, N));

    glm::vec3 V = glm::normalize(glm::cross(N, U));

    CameraRotateTrans = glm::mat4(glm::vec4(U[0], V[0], N[0], 0.0f),
                            glm::vec4(U[1], V[1], N[1], 0.0f),
                            glm::vec4(U[2], V[2], N[2], 0.0f),
                            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    return CameraRotateTrans * CameraTranslation;
}