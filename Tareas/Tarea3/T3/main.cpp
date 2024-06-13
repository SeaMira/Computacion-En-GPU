#include<iostream>
#include"terrain.h"
#include"shader.h"
#include"camera.h"
#include"init.h"

glm::vec3 pos(0.0, 0.0, 100.0);

cl::Device device;
cl::Platform platform;
cl::CommandQueue queue;
cl::Context context;
cl::Program program;
cl::Kernel kernel;

int height 100, width = 100, LOCAL_SIZE, GROUP_SIZE;
const std::string vertex_shader_path = "vertexShader.txt"
const std::string fragment_shader_path = "fragmentShader.txt"
const std::string terrain_path = "terrain.txt"

class TerrainSetup {
    private:
        Terrain* terrain;
        Camera* camera;
        GLuint terrainVao = -1;
        GLuint terrainVbo = -1;
        GLuint terrainIbo = -1;

        cl::BufferGL vertexBuff;
        cl::BufferGL texBuff;


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

        CreateTerrainVAO() {
            glGenVertexArrays(1, &terrainVao);
            glBindVertexArray(terrainVao);

            glGenBuffers(1, &terrainVbo);
            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)* terrain->verticesSize(), terrain->getVerticesData(), GL_STATIC_DRAW);

            // position
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);

            // tex coords
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));

            glGenBuffers(1, &terrainIbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle)* terrain->trianglesSize(), terrain->getTrianglesData(), GL_STATIC_DRAW);

            glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

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
}



int main(int argc, char const *argv[]) {

    Shader shaderObj(vertex_shader_path, fragment_shader_path);
    shaderObj.use();
   
    Terrain terrain(width, height);
    terrain.generateRandomTerrain(terrain_path);
    terrain.loadFromFile(terrain_path);

    Camara camera(pos, 100.0, 100.0, 100.0, glm::vec3(0.0, 0.0, 1.0));

    TerrainSetup GLTerrain(&terrain, &camera);







    GLFWwindow* window;
    initOpenGL(&window);

    initOpenCL(&device, &context, &platform);
    std::string src_code = load_from_file("kernel.cl");
    initProgram(&program, &kernel, src_code, &device, &queue, &context, src_code);

    GLTerrain.CreateTerrainVAO();
    glClearColor(1.0, 1.0, 1.0, 1.0);

    float lastFrameTime = glfwGetTime();
    float currentFrameTime;
    float deltaTime;

    while (!glfwWindowShouldClose(window)) {
        currentFrameTime = glfwGetTime();
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        updatePos(deltaTime/10);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(ShaderProgram);
        animate(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

