// #include <GL/glew.h>
// #include <glad/glad.h>
// #include <GLFW/glfw3.h>

#include"init.h"
#include"shader.h"
#include<iostream>
#include<terrain.h>
#include"camera.h"



int height = 100, width = 100;

cl::Device device;
cl::Platform platform;
cl::CommandQueue queue;
cl::Context context;
cl::Program program;
cl::Kernel kernel;

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
        GLuint gWVPLocation;

        GLuint terrainVao = -1;
        GLuint terrainVbo = -1;
        GLuint terrainIbo = -1;

        cl::BufferGL vertexBuff;
        cl::BufferGL texBuff;

        int LOCAL_SIZE, GROUP_SIZE;

    public:
        TerrainSetup(Terrain* terrain, Camera* camera) {
            this->terrain = terrain;
            this->camera = camera;
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
                camera->key_callback(-dt, 0.0);
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                camera->key_callback(dt, 0.0);
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                camera->key_callback(0.0, dt);
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                camera->key_callback(0.0, -dt);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                camera->rotating_key_callback(-dt, 0.0);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                camera->rotating_key_callback(dt, 0.0);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                camera->rotating_key_callback(0.0, dt);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                camera->rotating_key_callback(0.0, -dt);
             
        }

        void CreateTerrainVAO() {
            std::cout << "Creating VAO" << std::endl;
            std::cout << "Vertices count: " << terrain->verticesSize() << std::endl;
            std::cout << "Size of Vertex: " << sizeof(Vertex) << std::endl;
            std::cout << "Indexes count: " << terrain->trianglesSize() << std::endl;
            std::cout << "Size of Triangle: " << sizeof(Triangle) << std::endl;
            glGenVertexArrays(1, &terrainVao);
            glBindVertexArray(terrainVao);

            glGenBuffers(1, &terrainVbo);
            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*terrain->verticesSize(), terrain->getVerticesData(), GL_STATIC_DRAW);

            // Vertex* vertices = terrain->getVerticesData();
            // for (int i = 0; i < terrain->verticesSize(); i++) {
            //     std::cout << vertices->x << std::endl;
            //     vertices ++;
            // }
            glGenBuffers(1, &terrainIbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle)*terrain->trianglesSize(), terrain->getTrianglesData(), GL_STATIC_DRAW);
            

            cl_int err;
            vertexBuff = cl::BufferGL(context, CL_MEM_READ_WRITE, terrainVbo, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL buffer from OpenGL buffer" << std::endl;
                exit(EXIT_FAILURE);
            }
            
            texBuff = cl::BufferGL(context, CL_MEM_READ_WRITE, terrainIbo, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL buffer from OpenGL buffer" << std::endl;
                exit(EXIT_FAILURE);
            }
            std::cout << "Buffers CL: " << err << std::endl;
            
        }

        void camLoc(Shader* shader) {
            const std::string camLoc = "gWVP";
            gWVPLocation = shader->get(camLoc);
            if (gWVPLocation == -1) {
                printf("Error getting uniform location of 'gWVP'\n");
                exit(1);
            }
        }

        void RenderTerrain(GLFWwindow* window) {
            // for (int i = 0; i < 4; i++) {
            //     for (int j = 0; j < 4; j++) {
            //         std::cout << camera->view()[i][j] << std::endl;
            //     }
            // }            

            glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, (GLfloat*)(&(camera->view()[0][0])));
            // std::cout << "Posicion camara asignada" << std::endl;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glBindVertexArray(terrainVao);
            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);

            // position
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
            // std::cout << "Posicion vertices" << std::endl;

            // tex coords
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
            // std::cout << "Colores vertices" << std::endl;

            glDrawElements(GL_TRIANGLES, (terrain->trianglesSize())*3, GL_UNSIGNED_INT, 0);
            
            checkGLErrors("After RenderTerrain");
            // glBindVertexArray(0);
            // glDisableVertexAttribArray(0);
            // glDisableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            // std::cout << "Rendered frame." << std::endl;
            glfwSwapBuffers(window);
            glfwPollEvents();
            checkGLErrors("After PollEvents");
        }

        void updatePos(float dt) {
            cl::Event ev;
            glFinish();
            // Acquiring OpenGL objects in OpenCL
            std::vector<cl::Memory> glObjects = {vertexBuff, texBuff};
            cl_int res = queue.enqueueAcquireGLObjects(&glObjects, NULL, &ev);
            ev.wait();
            // std::cout<<5<<std::endl;
            if (res!=CL_SUCCESS) {
                std::cout<<"Failed acquiring GL object: "<<res<<std::endl;
                exit(248);
            }

            // float step = 0.0001f;
            // Set the kernel arguments
            kernel.setArg(0, vertexBuff);
            kernel.setArg(1, texBuff);
            kernel.setArg(2, dt);
            // kernel.setArg(2, NUM_PARTICLES);
            cl::NDRange GlobalWorkSize(GROUP_SIZE, 1, 1);
            cl::NDRange LocalWorkSize(LOCAL_SIZE, 1, 1);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, GlobalWorkSize, LocalWorkSize);
            
            res = queue.enqueueReleaseGLObjects(&glObjects);
            if (res!=CL_SUCCESS) {
                std::cout<<"Failed releasing GL object: "<<res<<std::endl;
                exit(247);
            }

            queue.finish();
        }
};



int main(int argc, char const *argv[]) {
    GLFWwindow* window;
    initOpenGL(&window);

    Shader shaderObj(vertex_shader_path, fragment_shader_path);
    shaderObj.use();
   
   std::cout << "shaderObj" << std::endl;
    Terrain terrain(width, height);
   std::cout << "terrain" << std::endl;
    terrain.generateRandomTerrain(terrain_path);
   std::cout << "generate terrain" << std::endl;
//     terrain.loadFromFile(terrain_path);
//    std::cout << "load terrain" << std::endl;

    glm::vec3 pos(50.0f, 50.0f, 150.0f);
    float cameraDistance = 100.0f; // Distancia de la cámara al terreno
    float cameraYaw = -90.0f; // Yaw para mirar hacia el centro del terreno
    float cameraPitch = -45.0f; // Pitch para mirar hacia abajo
    Camera camera(pos, cameraDistance, cameraYaw, cameraPitch, glm::vec3(0.0, 0.0, 1.0));
   std::cout << "camera" << std::endl;

    TerrainSetup GLTerrain(&terrain, &camera);
   std::cout << "terrain setup" << std::endl;
    GLTerrain.camLoc(&shaderObj);
   std::cout << "camera location" << std::endl;

    initOpenCL(&device, &context, &platform);

    std::string src_code = load_from_file("kernel.cl");
    std::string kernel_name = "terrainManipulation";
    initProgram(&program, &kernel, src_code, &device, &queue, &context, kernel_name);

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);

    GLTerrain.CreateTerrainVAO();
    glClearColor(0.0, 0.0, 0.0, 0.0);

    float lastFrameTime = glfwGetTime();
    float currentFrameTime;
    float deltaTime;

    while (!glfwWindowShouldClose(window)) {
        currentFrameTime = glfwGetTime();
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        // updatePos(deltaTime/10);
        // shaderObj.use();
        GLTerrain.processInput(window, deltaTime*10.0f);
        GLTerrain.RenderTerrain(window);
        
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

