#include "init.h"


void error_callback(int error, const char* description) {
    std::cerr << "Error: " << description << std::endl;
}

void initOpenGL(GLFWwindow** window) {
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    // glfwWindowHint(GLFW_RED_BITS, 8);
    // glfwWindowHint(GLFW_GREEN_BITS, 8);
    // glfwWindowHint(GLFW_BLUE_BITS, 8);
    // glfwWindowHint(GLFW_ALPHA_BITS, 8);
    // glfwWindowHint(GLFW_DEPTH_BITS, 24);
    // glfwWindowHint(GLFW_STENCIL_BITS, 8);

    *window = glfwCreateWindow(640, 480, "OpenGL & OpenCL Interoperability", NULL, NULL);
    if (!*window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    glfwMakeContextCurrent(*window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
}