#include <setup.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
// void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window, float deltaTime);

#define MAX_CURVES 150
// settings
unsigned int SCR_WIDTH;
unsigned int SCR_HEIGHT;
int GRID_SIZE;
int curves = 0; // Inicializar el contador de curvas

// camera
Camera* globCamera;

// terrain
Terrain* globTerrain;

// setup
TerrainSetup* setup;

// point lights default settings
PointLights* pointLightsSetup;
glm::vec3 defAmbient(0.05f, 0.05f, 0.05f);
glm::vec3 defDiffuse(0.8f, 0.8f, 0.8f);
glm::vec3 defSpecular(1.0f, 1.0f, 1.0f);
float defConstant = 1.0f;
float defLinear = 0.09f;
float defQuadratic = 0.032f;

// directional light default settings
glm::vec3 defDirAmbient(0.05f, 0.05f, 0.05f);
glm::vec3 defDirDiffuse(0.4f, 0.4f, 0.4f);
glm::vec3 defDirSpecular(0.5f, 0.5f, 0.5f);

// general illumination settigns
glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
glm::vec3 objectColor(1.0f, 0.5f, 0.31f);

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

float curve_heights[MAX_CURVES];
float d0 = 0.0f;

void addTerrainPointLight(PointLights* pointLights, float height);
void addTerrainDirLight(Shader* shader, glm::vec3 pos);

void CameraInfo() {
    ImGui::Begin("Camera Info");   
    ImGui::Text("Position (%.2f, %.2f, %.2f)", globCamera->getPosition().x, globCamera->getPosition().y, globCamera->getPosition().z);
    ImGui::InputFloat("input new height", &d0, 0.01f, 1.0f, "%.2f");
    if (ImGui::Button("Add curve height")) {
        curve_heights[curves] = d0;
        curves++;
    }
    if (ImGui::Button("Add Light")) {
        addTerrainPointLight(pointLightsSetup, globTerrain->maxHeight);
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Light")) {
        pointLightsSetup->removePointLight();
    }
    ImGui::End();
}

void addTerrainPointLight(PointLights* pointLights, float height) {
    int lts = pointLights->nr_lights;
    if (lts < 39) {
        float terrDt = (float)GRID_SIZE/8.0f;
        int x = (lts+1)/8;
        int y = lts+1-8*x;
        glm::vec3 pos((float)x*terrDt, (float)y*terrDt, height);

        PointLight newPointLight(pos, defAmbient, defDiffuse, defSpecular, 
            defConstant, defLinear, defQuadratic);
        pointLights->addPointLight(newPointLight);

    } 
}

void addTerrainDirLight(Shader* shader, glm::vec3 pos) {
    shader->setVec3("dirLight.direction", pos.x, pos.y, pos.z);
    shader->setVec3("dirLight.ambient", defDirAmbient.x, defDirAmbient.y, defDirAmbient.z);
    shader->setVec3("dirLight.diffuse", defDirDiffuse.x, defDirDiffuse.y, defDirDiffuse.z);
    shader->setVec3("dirLight.specular", defDirSpecular.x, defDirSpecular.y, defDirSpecular.z);
}


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
    
    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    Interface gui(window);
    gui.initContext();

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    // glfwSetScrollCallback(window, scroll_callback);

    gui.initWithOpenGL();

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader program
    // ------------------------------------
    Shader ourShader("vertexShader.txt", "fragmentShader.txt");

    PointLights pointLights(&ourShader);
    pointLightsSetup = &pointLights;

    Shader geomShader("vertexShaderGEOM.txt", "fragmentShaderGEOM.txt", "geometryShader.txt");


    Terrain terrain(GRID_SIZE, roughness);
    globTerrain = &terrain;
    terrain.generateRandomTerrain("terrain.txt");
    TerrainSetup GLTerrain(&terrain);
    setup = &GLTerrain;
    GLTerrain.CreateTerrainVAO();    

    glLineWidth(3); 
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    // -------------------------------------------------------------------------------------------
    ourShader.use();
    
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    camera.SetPosition((float)GRID_SIZE/2, (float)GRID_SIZE/2, terrain.maxHeight);
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

        gui.newFrame();


        // activate shader
        ourShader.use();
        glm::vec3 lightPos((float)GRID_SIZE/2, (float)GRID_SIZE/2, terrain.maxHeight);

        // camera/view transformation
        ourShader.setMat4("projection", camera.getProjection() );
        ourShader.setMat4("view", camera.getView());
        ourShader.setMat4("model", camera.getModel());
        ourShader.setFloat("minHeight", terrain.minHeight);
        ourShader.setFloat("maxHeight", terrain.maxHeight);

        // illumination
        addTerrainDirLight(&ourShader, glm::vec3(0.0f, 0.0f, -1.0f));
        pointLights.setPointLights();
        ourShader.setVec3("objectColor", objectColor.x, objectColor.y, objectColor.z);
        ourShader.setVec3("lightColor", lightColor.x, lightColor.y, lightColor.z);
        ourShader.setVec3("viewPos", camera.getPosition());

        GLTerrain.RenderTerrain(deltaTime*10.0f);

        
        geomShader.use();

        geomShader.setMat4("projection", camera.getProjection() );
        geomShader.setMat4("view", camera.getView());
        geomShader.setMat4("model", camera.getModel());
        geomShader.setFloat("minHeight", terrain.minHeight );
        geomShader.setFloat("maxHeight", terrain.maxHeight );
        geomShader.setInt("curves", curves);
        for (int i = 0; i < curves; i++) {
            std::stringstream curvePos;
            curvePos << "curve_heights[" << i << "]";
            geomShader.setFloat(curvePos.str().c_str(), curve_heights[i]);
        }
       
        GLTerrain.RenderTerrain(deltaTime*10.0f);


        CameraInfo();
        gui.drawData();

        
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
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
        setup->reGenerateRandomTerrain();
        curves = 0;    
    }


    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
        addTerrainPointLight(pointLightsSetup, globTerrain->maxHeight);
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
        pointLightsSetup->removePointLight();
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