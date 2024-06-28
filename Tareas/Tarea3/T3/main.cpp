#include <setup.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
// void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window, float deltaTime);

// settings
unsigned int SCR_WIDTH;
unsigned int SCR_HEIGHT;
int GRID_SIZE;
int curves = 1; // Inicializar el contador de curvas

// camera
Camera* globCamera;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;




int main(int argc, char const *argv[]) {

    GRID_SIZE = std::stoi(argv[1]);
    SCR_WIDTH = std::stoi(argv[2]);
    SCR_HEIGHT = std::stoi(argv[3]);
    float roughness = std::stof(argv[4]); 
    std::cout << "inputs <GRID_SIZE> <SCR_WIDTH> <SCR_HEIGHT> <roughness>" << std::endl;

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    // glfwSetScrollCallback(window, scroll_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader program
    // ------------------------------------
    Shader ourShader("vertexShader.txt", "fragmentShader.txt");
    Shader geomShader("vertexShaderGEOM.txt", "fragmentShaderGEOM.txt", "geometryShader.txt");


    Terrain terrain(GRID_SIZE, roughness);
    terrain.generateRandomTerrain("terrain.txt");
    TerrainSetup GLTerrain(&terrain);
    GLTerrain.CreateTerrainVAO();    

    glLineWidth(3); 
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    // -------------------------------------------------------------------------------------------
    ourShader.use();
    
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    globCamera = &camera;

    // pass projection matrix to shader (as projection matrix rarely changes there's no need to do this per frame)
    // -----------------------------------------------------------------------------------------------------------
    // float contourHeight = (terrain.maxHeight + terrain.minHeight)/2;
    // std::cout << "contour height: " << contourHeight << std::endl;


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window, deltaTime);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        // bind textures on corresponding texture units
        // glActiveTexture(GL_TEXTURE0);
        // glBindTexture(GL_TEXTURE_2D, texture1);
        // glActiveTexture(GL_TEXTURE1);
        // glBindTexture(GL_TEXTURE_2D, texture2);

        // activate shader
        ourShader.use();
        glm::vec3 lightPos((float)GRID_SIZE/2, (float)GRID_SIZE/2, terrain.maxHeight);

        // camera/view transformation
        ourShader.setMat4("projection", camera.getProjection() );
        ourShader.setMat4("view", camera.getView());
        ourShader.setMat4("model", camera.getModel());
        ourShader.setFloat("minHeight", terrain.minHeight);
        ourShader.setFloat("maxHeight", terrain.maxHeight);
        ourShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        ourShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
        ourShader.setVec3("lightPos", lightPos);
        ourShader.setVec3("viewPos", camera.getPosition());

        GLTerrain.RenderTerrain(deltaTime*10.0f);

        
        geomShader.use();

        geomShader.setMat4("projection", camera.getProjection() );
        geomShader.setMat4("view", camera.getView());
        geomShader.setMat4("model", camera.getModel());
        // geomShader.setFloat("height", contourHeight );
        geomShader.setInt("curves", curves);
        geomShader.setFloat("minHeight", terrain.minHeight);
        geomShader.setFloat("maxHeight", terrain.maxHeight);
        GLTerrain.RenderTerrain(deltaTime*10.0f);
        // render boxes
        // glBindVertexArray(VAO);
        // for (unsigned int i = 0; i < 10; i++)
        // {
        //     // calculate the model matrix for each object and pass it to shader before drawing
        //     glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        //     model = glm::translate(model, cubePositions[i]);
        //     float angle = 20.0f * i;
        //     model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
        //     ourShader.setMat4("model", model);

        //     glDrawArrays(GL_TRIANGLES, 0, 36);
        // }
        camera.OnRender(deltaTime*10.0f);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        globCamera->OnKeyboard(1, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        globCamera->OnKeyboard(2, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        globCamera->OnKeyboard(3, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        globCamera->OnKeyboard(4, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        globCamera->OnKeyboard(5, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        globCamera->OnKeyboard(6, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        curves++;
        std::cout << "Curves increased to: " << curves << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        if (curves > 1) {
            curves--;
            std::cout << "Curves decreased to: " << curves << std::endl;
        }
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    globCamera->OnMouse((float)xposIn, (float)yposIn);
}

// // glfw: whenever the mouse scroll wheel scrolls, this callback is called
// // ----------------------------------------------------------------------
// void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
// {
//     fov -= (float)yoffset;
//     if (fov < 1.0f)
//         fov = 1.0f;
//     if (fov > 45.0f)
//         fov = 45.0f;
// }