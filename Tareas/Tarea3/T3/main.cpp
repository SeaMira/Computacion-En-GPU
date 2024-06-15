// #include <GL/glew.h>
// #include <glad/glad.h>
// #include <GLFW/glfw3.h>

#include"init.h"
#include"shader.h"
#include<iostream>
#include<terrain.h>
#include"camera2.h"

Camera* globCamera;

int height = 10, width = 10;

// cl::Device device;
// cl::Platform platform;
// cl::CommandQueue queue;
// cl::Context context;
// cl::Program program;
// cl::Kernel kernel;

const std::string vertex_shader_path = "vertexShader.txt";
const std::string fragment_shader_path = "fragmentShader.txt";
const std::string terrain_path = "terrain.txt";

void checkGLErrors(const std::string& location) {
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error at " << location << ": " << error << std::endl;
    }
}

class TerrainSetup {
    private:
        Terrain* terrain;
        Camera* camera;
        WorldTrans* wrldTrans;
        ProjectionTrans* projInfo;
        glm::mat4 mat;

        GLuint gWVPLocation;

        GLuint terrainVao = -1;
        GLuint terrainVbo = -1;
        GLuint terrainIbo = -1;

        cl::BufferGL vertexBuff;
        cl::BufferGL texBuff;

        int LOCAL_SIZE, GROUP_SIZE;

    public:
        TerrainSetup(Terrain* terrain, Camera* camera, WorldTrans* wrldTrans, ProjectionTrans* projInfo) {
            this->terrain = terrain;
            this->camera = camera;
            this->wrldTrans = wrldTrans;
            this->projInfo = projInfo;

            mat = (projInfo->getProjectionMatrix()) *(camera->GetMatrix()) * (wrldTrans->GetMatrix());
        }

        void updateMat() {
            std::cout << "Projection Matrix: " << std::endl;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    std::cout << projInfo->getProjectionMatrix()[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "View Matrix: " << std::endl;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    std::cout << camera->GetMatrix()[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "World Matrix: " << std::endl;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    std::cout << wrldTrans->GetMatrix()[i][j] << " ";
                }
                std::cout << std::endl;
            }
            mat = (projInfo->getProjectionMatrix()) *(camera->GetMatrix()) * (wrldTrans->GetMatrix());
        }

        ~TerrainSetup() {
            if (terrain) {
                delete terrain;
            }
            if (camera) {
                delete camera;
            }
            if (terrainVao != -1) {
                glDeleteBuffers(1, &terrainVao);
            }
            if (terrainVbo != -1) {
                glDeleteBuffers(1, &terrainVbo);
            }
            if (terrainIbo != -1) {
                glDeleteBuffers(1, &terrainIbo);
            }
        }

        void processInput(GLFWwindow* window, float dt) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);

            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
                camera->OnKeyboard(2, dt);
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                camera->OnKeyboard(3, dt);
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                camera->OnKeyboard(0, dt);
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                camera->OnKeyboard(1, dt);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                camera->OnKeyboard(7, dt);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                camera->OnKeyboard(6, dt);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                camera->OnKeyboard(4, dt);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                camera->OnKeyboard(5, dt);
            
        }

        void CreateTerrainVAO() {
            std::cout << "Creating VAO" << std::endl;
            std::cout << "Vertices count: " << terrain->verticesSize() << std::endl;
            std::cout << "Size of Vertex: " << sizeof(Vertex) << std::endl;
            std::cout << "Indexes count: " << terrain->trianglesSize() << std::endl;
            std::cout << "Size of Triangle: " << sizeof(Triangle) << std::endl;
            std::cout << "Size of unsigned int: " << sizeof(unsigned int) << std::endl;
            glGenVertexArrays(1, &terrainVao);
            glBindVertexArray(terrainVao);

            glGenBuffers(1, &terrainVbo);
            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*terrain->verticesSize(), terrain->getVerticesData(), GL_STATIC_DRAW);

            // Vertex* vertices = terrain->getVerticesData();
            // for (int i = 0; i < terrain->verticesSize(); i++) {
            //     std::cout << vertices->x << " " << vertices->y << " " << vertices->z << std::endl;
            //     vertices ++;
            // }
            glGenBuffers(1, &terrainIbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle)*terrain->trianglesSize(), terrain->getTrianglesData(), GL_STATIC_DRAW);
            // Triangle* triangles = terrain->getTrianglesData();
            // for (int i = 0; i < terrain->trianglesSize(); i++) {
            //     std::cout << triangles->v1 << " " << triangles->v2 << " " << triangles->v3 << std::endl;
            //     triangles ++;
            // }

            // cl_int err;
            // vertexBuff = cl::BufferGL(context, CL_MEM_READ_WRITE, terrainVbo, &err);
            // if (err != CL_SUCCESS) {
            //     std::cerr << "Failed to create OpenCL buffer from OpenGL buffer" << std::endl;
            //     exit(EXIT_FAILURE);
            // }
            
            // texBuff = cl::BufferGL(context, CL_MEM_READ_WRITE, terrainIbo, &err);
            // if (err != CL_SUCCESS) {
            //     std::cerr << "Failed to create OpenCL buffer from OpenGL buffer" << std::endl;
            //     exit(EXIT_FAILURE);
            // }
            // std::cout << "Buffers CL: " << err << std::endl;
            
        }

        void camLoc(Shader* shader) {
            const std::string camLoc = "gWVP";
            gWVPLocation = shader->get(camLoc);
            if (gWVPLocation == -1) {
                printf("Error getting uniform location of 'gWVP'\n");
                exit(1);
            }
        }

        void RenderTerrain(GLFWwindow* window, float dt) {
            // for (int i = 0; i < 4; i++) {
            //     for (int j = 0; j < 4; j++) {
            //         std::cout << camera->view()[i][j] << std::endl;
            //     }
            // }            
            // wrldTrans->Rotate(0.03f, 0.02f, 0.01f);
            // camera->OnRender(dt);
            // std::cout << camera->m_pos[0] << " " << camera->m_pos[1] << " " << camera->m_pos[2] << std::endl;
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            updateMat();

            glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, glm::value_ptr(mat));
            // std::cout << "Posicion camara asignada" << std::endl;
            glBindVertexArray(terrainVao);

            // glBindVertexArray(terrainVao);
            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);

            // position
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
            // std::cout << "Posicion vertices" << std::endl;

            // color coords
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
            // std::cout << "Colores vertices" << std::endl;
            glLineWidth(1); 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_TRIANGLES, (terrain->trianglesSize())*3, GL_UNSIGNED_INT, 0);
            
            checkGLErrors("After RenderTerrain");
            // glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            // glBindBuffer(GL_ARRAY_BUFFER, 0);
            // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            // std::cout << "Rendered frame." << std::endl;
            glfwSwapBuffers(window);
            glfwPollEvents();
            checkGLErrors("After PollEvents");
        }

        // void updatePos(float dt) {
        //     cl::Event ev;
        //     glFinish();
        //     // Acquiring OpenGL objects in OpenCL
        //     std::vector<cl::Memory> glObjects = {vertexBuff, texBuff};
        //     cl_int res = queue.enqueueAcquireGLObjects(&glObjects, NULL, &ev);
        //     ev.wait();
        //     // std::cout<<5<<std::endl;
        //     if (res!=CL_SUCCESS) {
        //         std::cout<<"Failed acquiring GL object: "<<res<<std::endl;
        //         exit(248);
        //     }

        //     // float step = 0.0001f;
        //     // Set the kernel arguments
        //     kernel.setArg(0, vertexBuff);
        //     kernel.setArg(1, texBuff);
        //     kernel.setArg(2, dt);
        //     // kernel.setArg(2, NUM_PARTICLES);
        //     cl::NDRange GlobalWorkSize(GROUP_SIZE, 1, 1);
        //     cl::NDRange LocalWorkSize(LOCAL_SIZE, 1, 1);

        //     queue.enqueueNDRangeKernel(kernel, cl::NullRange, GlobalWorkSize, LocalWorkSize);
            
        //     res = queue.enqueueReleaseGLObjects(&glObjects);
        //     if (res!=CL_SUCCESS) {
        //         std::cout<<"Failed releasing GL object: "<<res<<std::endl;
        //         exit(247);
        //     }

        //     queue.finish();
        // }
};

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (globCamera) {
        globCamera->OnMouse(static_cast<int>(xpos), static_cast<int>(ypos));
    }
}

int main(int argc, char const *argv[]) {
    GLFWwindow* window;
    initOpenGL(&window);

   
   std::cout << "shaderObj" << std::endl;
    Terrain terrain(width, height);
   std::cout << "terrain" << std::endl;
    terrain.generateRandomTerrain(terrain_path);
   std::cout << "generate terrain" << std::endl;
//     terrain.loadFromFile(terrain_path);
//    std::cout << "load terrain" << std::endl;

    // glm::vec3 pos(50.0f, 50.0f, 150.0f);
    // float cameraDistance = 100.0f; // Distancia de la cámara al terreno
    // float cameraYaw = -90.0f; // Yaw para mirar hacia el centro del terreno
    // float cameraPitch = -45.0f; // Pitch para mirar hacia abajo
    // Camera camera(pos, cameraDistance, cameraYaw, cameraPitch, glm::vec3(0.0, 0.0, 1.0));
    Camera camera(640, 480);
    globCamera = &camera;
    camera.SetPosition((float)width, (float)height, 10.0f); // Posición inicial de la cámara
    // camera.SetOrientation(0.0f, 0.0f); // Orientación inicial de la cámara

    ProjectionTrans persProj(640, 480);

    WorldTrans worldtr;
    // worldtr.SetPosition((float)width, (float)height, 0.0f); // El objeto está centrado en el origen
    // worldtr.SetScale(1.0f); // Escala unitaria
    

   std::cout << "camera" << std::endl;

    TerrainSetup GLTerrain(&terrain, &camera, &worldtr, &persProj);
   std::cout << "terrain setup" << std::endl;
   std::cout << "camera location" << std::endl;

    // initOpenCL(&device, &context, &platform);

    // std::string src_code = load_from_file("kernel.cl");
    // std::string kernel_name = "terrainManipulation";
    // initProgram(&program, &kernel, src_code, &device, &queue, &context, kernel_name);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    // glfwSetCursorPosCallback(window, mouse_callback);
    // glFrontFace(GL_CW);
    // glCullFace(GL_BACK);

    GLTerrain.CreateTerrainVAO();
    Shader shaderObj(vertex_shader_path, fragment_shader_path);
    shaderObj.use();
    GLTerrain.camLoc(&shaderObj);

    float lastFrameTime = glfwGetTime();
    float currentFrameTime;
    float deltaTime;

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0, 0.0, 0.0, 0.0);
        currentFrameTime = glfwGetTime();
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        // updatePos(deltaTime/10);
        shaderObj.use();
        GLTerrain.processInput(window, deltaTime*10.0f);
        GLTerrain.RenderTerrain(window, deltaTime*10.0f);
        
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

