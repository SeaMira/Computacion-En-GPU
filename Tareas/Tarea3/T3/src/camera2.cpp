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
    // // Matriz de Escalado
    // glm::mat4 Scale = glm::scale(glm::mat4(1.0f), glm::vec3(m_scale, m_scale, m_scale));

    // // Matriz de Rotación
    // glm::mat4 Rotation = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    // Rotation = glm::rotate(Rotation, glm::radians(m_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    // Rotation = glm::rotate(Rotation, glm::radians(m_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    // // Matriz de Traslación
    // glm::mat4 Translation = glm::translate(glm::mat4(1.0f), glm::vec3(m_pos.x, m_pos.y, m_pos.z));

    // glm::mat4 WorldTransformation = Translation * Rotation * Scale;

    // return WorldTransformation;
    return glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0))*glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
}


ProjectionTrans::ProjectionTrans(int width, int height) {
    this->width = width;
    this->height = height;
}

glm::mat4 ProjectionTrans::getProjectionMatrix() {
    float ar = static_cast<float>(width) / static_cast<float>(height);
    return glm::perspective(glm::radians(FOV), ar, zNear, zFar);
    // glm::mat4 Projection = glm::perspectiveFovLH(glm::radians(FOV), width, height, zNear, zFar);
}



static int MARGIN = 30;
static float EDGE_STEP = 1.0f;

Camera::Camera() {
    m_pos          = glm::vec3(50.0f, 50.0f, 40.0f);
    m_front        = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up           = glm::vec3(0.0f, 1.0f, 0.0f);

    m_windowWidth = 640;
    m_windowHeight = 480;
    Init();
}

Camera::Camera(int WindowWidth, int WindowHeight) {
    m_pos          = glm::vec3(1.0f, 1.0f, 5.0f);
    m_front       = glm::vec3(0.0f, 0.0f, -1.0f);
    m_up           = glm::vec3(0.0f, 1.0f, 0.0f);
    m_windowWidth = WindowWidth;
    m_windowHeight = WindowHeight;
    Init();
}

Camera::Camera(int WindowWidth, int WindowHeight, const glm::vec3& Pos, const glm::vec3& Front, const glm::vec3& Up) {
    m_pos          = Pos;
    m_front        = glm::normalize(Front);
    m_up           = glm::normalize(Up);
    m_windowWidth = WindowWidth;
    m_windowHeight = WindowHeight;
    Init();
}

void Camera::Init()
{
    glm::vec3 HFront(m_front.x, 0.0, m_front.z);
    HFront = glm::normalize(HFront);

    float Angle = glm::degrees(asin(abs(HFront.z)));

    if (HFront.z >= 0.0f)
    {
        if (HFront.x >= 0.0f)
        {
            m_AngleH = 360.0f - Angle;
        }
        else
        {
            m_AngleH = 180.0f + Angle;
        }
    }
    else
    {
        if (HFront.x >= 0.0f)
        {
            m_AngleH = Angle;
        }
        else
        {
            m_AngleH = 180.0f - Angle;
        }
    }

    m_AngleV = -glm::degrees(asin(m_front.y));

    m_OnUpperEdge = false;
    m_OnLowerEdge = false;
    m_OnLeftEdge  = false;
    m_OnRightEdge = false;
    m_mousePos.x  = m_windowWidth / 2;
    m_mousePos.y  = m_windowHeight / 2;
}


void Camera::SetPosition(float x, float y, float z) {
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = z;
    Update();
}

void Camera::SetOrientation(float AngleH, float AngleV) {
    this->m_AngleH = m_AngleH;
    this->m_AngleV = AngleV;
    Update();
}

// void Camera::updateCameraVectors() {
//     // Calcula el nuevo vector de frente
//     glm::vec3 front;
//     front.x = cos(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));
//     front.y = sin(glm::radians(m_AngleV));
//     front.z = sin(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));
//     m_front = glm::normalize(front);
//     // También re-calcula el vector de la derecha y el de arriba
//     m_right = glm::normalize(glm::cross(m_front, m_up)); // Normaliza el vector de la derecha
//     m_up = glm::normalize(glm::cross(m_right, m_front));
//     // m_up está predefinido como (0.0f, 1.0f, 0.0f)
// }


void Camera::OnKeyboard(int key, float dt) {
    switch (key) {
    // up
    case 0:
        m_pos += (m_front * m_speed*dt);
        break;
    // down
    case 1:
        m_pos -= (m_front * m_speed*dt);
        break;
    // left
    case 2:
        {
            glm::vec3 Left = glm::cross(m_front,m_up);
            Left = glm::normalize(Left);
            Left *= m_speed*dt;
            // m_front -= Left;
            m_pos -= Left;
        }
        break;
    // right
    case 3:
        {
            glm::vec3 Right = glm::cross(m_up, m_front);
            Right = glm::normalize(Right);
            Right *= m_speed*dt;
            // m_front -= Right;
            m_pos -= Right;
        }
        break;
    // w
    case 4:
        m_pos.y += m_speed*dt;
        break;
    // s
    case 5:
        m_pos.y -= m_speed*dt;
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
    // Update();
    
}

void Camera::OnMouse(int x, int y)
{
    int DeltaX = x - m_mousePos.x;
    int DeltaY = y - m_mousePos.y;

    m_mousePos.x = x;
    m_mousePos.y = y;

    m_AngleH += (float)DeltaX / 20.0f;
    m_AngleV += (float)DeltaY / 50.0f;

    if (abs(DeltaX) <= MARGIN && x <= MARGIN) {
        m_OnLeftEdge = true;
        m_OnRightEdge = false;
    }
    else if (abs(DeltaX) <= MARGIN && x >= (m_windowWidth - MARGIN)) {
        m_OnLeftEdge = false;
        m_OnRightEdge = true;
    }
    else {
        m_OnLeftEdge = false;
        m_OnRightEdge = false;
    }

    if (abs(DeltaY) <= MARGIN && y <= MARGIN) {
        m_OnUpperEdge = true;
        m_OnLowerEdge = false;
    }
    else if (abs(DeltaY) <= MARGIN && y >= (m_windowHeight - MARGIN)) {
        m_OnUpperEdge = false;
        m_OnLowerEdge = true;
    }
    else {
        m_OnUpperEdge = false;
        m_OnLowerEdge = false;
    }

    Update();
}

glm::vec3 RotateVector(const glm::vec3& vec, float angle, const glm::vec3& axis) {
    glm::quat rotationQuat = glm::angleAxis(angle, glm::normalize(axis));
    glm::quat conjugateQuat = glm::conjugate(rotationQuat);
    glm::quat w = rotationQuat * glm::quat(0, vec) * conjugateQuat;

    return glm::vec3(w.x, w.y, w.z);
}

void Camera::Update(){
    
    glm::vec3 Yaxis(0.0f, 1.0f, 0.0f);

    // Inicializar el vector de vista
    // glm::vec3 View(1.0f, 0.0f, 0.0f);
    glm::vec3 View;

    View.x = cos(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));
    View.y = sin(glm::radians(m_AngleV));
    View.z = sin(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));
    View = glm::normalize(View);

    // Rotar el vector de vista por el ángulo horizontal alrededor del eje vertical
    // View = RotateVector(View, glm::radians(m_AngleH), Yaxis);
    // View = glm::normalize(View);

    // // Obtener el vector horizontal (U) como el producto cruzado del eje Y y el vector de vista
    glm::vec3 U = glm::cross(Yaxis, View);
    U = glm::normalize(U);

    // // Rotar el vector de vista por el ángulo vertical alrededor del eje horizontal
    // View = RotateVector(View, glm::radians(m_AngleV), U);
    // View = glm::normalize(View);

    // Establecer m_front y m_up
    m_front = View;
    m_up = glm::normalize(glm::cross(U, m_front));
}

void Camera::OnRender(float dt)
{
    this->dt = dt;
    bool ShouldUpdate = false;

    if (m_OnLeftEdge) {
        m_AngleH -= EDGE_STEP*dt;
        ShouldUpdate = true;
    }
    else if (m_OnRightEdge) {
        m_AngleH += EDGE_STEP*dt;
        ShouldUpdate = true;
    }

    if (m_OnUpperEdge) {
        if (m_AngleV > -90.0f) {
            m_AngleV -= EDGE_STEP*dt;
            ShouldUpdate = true;
        }
    }
    else if (m_OnLowerEdge) {
        if (m_AngleV < 90.0f) {
           m_AngleV += EDGE_STEP*dt;
           ShouldUpdate = true;
        }
    }

    if (ShouldUpdate) {
        Update();
    }
}


glm::mat4 Camera::GetMatrix() {
    // glm::mat4 CameraTranslation, CameraRotateTrans;
    // CameraTranslation = glm::mat4(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
    //                         glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
    //                         glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
    //                         glm::vec4(-m_pos[0], -m_pos[1], -m_pos[2], 1.0f));

    // glm::vec3 front;
    // front.x = cos(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));
    // front.y = sin(glm::radians(m_AngleV));
    // front.z = sin(glm::radians(m_AngleH)) * cos(glm::radians(m_AngleV));        
    // m_front = glm::normalize(front);

    // glm::vec3 N = m_front;
    // N = glm::normalize(N);
    // glm::vec3 upNorm = m_up;
    // upNorm = glm::normalize(upNorm);

    // glm::vec3 U = glm::normalize(glm::cross(upNorm, N));

    // glm::vec3 V = glm::normalize(glm::cross(N, U));

    // CameraRotateTrans = glm::mat4(glm::vec4(U[0], V[0], N[0], 0.0f),
    //                         glm::vec4(U[1], V[1], N[1], 0.0f),
    //                         glm::vec4(U[2], V[2], N[2], 0.0f),
    //                         glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    // return CameraRotateTrans * CameraTranslation;
    return glm::lookAt(m_pos, m_pos + m_front, m_up);
    // return glm::lookAt(glm::vec3(1.0, 1.0, 2.5), glm::vec3(1.0, 1.0,0), glm::vec3(0,1,0));
}