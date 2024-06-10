#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <device_launch_parameters.h>


// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// Utilities and timing functions
#include <cuda.h>    // includes cuda.h and cuda_runtime_api.h
#define     DIM    512

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;




__global__
void random_points(float* pos, unsigned int N, int rand)
{
//    int i = threadIdx.x + blockDim.x * blockIdx.x;
      int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        pos[6 * i + 0] = sin(pos[6 * i + 0] * rand + 10);
        pos[6 * i + 1] = sin(pos[6 * i + 1] * rand + 16);
        pos[6 * i + 2] = 0.0;
        pos[6 * i + 3] = 1;
        pos[6 * i + 4] = abs(pos[6 * i + 0]);
        pos[6 * i + 5] = abs(pos[6 * i + 1]);
    }
}



void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"out vec3 ourColor;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos, 1.0);\n"
"   ourColor = aColor;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(ourColor, 1.0f);\n"
"}\n\0";

int main()
{

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Test cuda opengl", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    srand(138);
    int N = 512;
    // initialize buffer object

    float* h_Mem = (float*)malloc(6 * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_Mem[6 * i + 0] = -1 + 2 * ((float)rand()) / RAND_MAX;
        h_Mem[6 * i + 1] = -1 + 2 * ((float)rand()) / RAND_MAX;
        h_Mem[6 * i + 2] = 0.0;
        h_Mem[6 * i + 3] = 1.0;
        h_Mem[6 * i + 4] = 1.0;
        h_Mem[6 * i + 5] = 1.0;
    }
    // create buffer object

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &vbo);

    //elijo VAO
    glBindVertexArray(VAO);

    //Copio elementos h_mem a memoria cuda
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * N * sizeof(float), h_Mem, GL_DYNAMIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, 6 * N * sizeof(float), NULL, GL_DYNAMIC_DRAW);


    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    float* dptr;
    size_t num_bytes;

    glClearColor(1.0, 1.0, 1.0, 1.0);


    // glUseProgram(shaderProgram);
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

    // map OpenGL buffer object for writing from CUDA
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource);
//      dim3 dimGrid(1, 1);
//      dim3 dimBlock(N, 1);
        int threadsPerBlock = 32;
        int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

        random_points <<<blocksPerGrid, threadsPerBlock >>> (dptr, N, rand());
        
// Para debugging
        cudaMemcpy(h_Mem, dptr, 6 * N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 6 * N; i++) {
            if (i % 6 == 0)
                printf("\n");
            printf("%.2f ", h_Mem[i]);
                      
        }
        printf("\n\n\n\n\n");
// Fin debugging

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0); 
        // render
        // ------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &VAO);


    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
