#include <setup.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window, float deltaTime);

#define MAX_CURVES 150

// settings
unsigned int SCR_WIDTH;
unsigned int SCR_HEIGHT;
int GRID_SIZE;
int curves = 0; // Inicializar el contador de curvas
bool useOrthoCamera = false;

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
float defLinear = 0.12f;
float defQuadratic = 0.0f;

// directional light default settings
glm::vec3 defDirDirection(0.05f, 0.05f, 0.05f);
glm::vec3 defDirAmbient(0.05f, 0.05f, 0.05f);
glm::vec3 defDirDiffuse(0.4f, 0.4f, 0.4f);
glm::vec3 defDirSpecular(0.5f, 0.5f, 0.5f);

// moon
PointLight moonLight(glm::vec3(0.0f, 0.0f, 5000.0f), glm::vec3(0.05f, 0.05f, 0.1f),
    glm::vec3(0.2f, 0.2f, 0.3f), glm::vec3(0.5f, 0.5f, 0.6f), 1.0f, 0.0f, 0.0f);

float dayCycleDuration = 60.0f; // Duración de un ciclo de día y noche en segundos
float currentTime = 0.0f;

// general illumination settigns
glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
glm::vec3 objectColor(1.0f, 0.5f, 0.31f);

// timing
float deltaTime = 0.0f;	
float lastFrame = 0.0f;

float curve_heights[MAX_CURVES];
float d0 = 0.0f;

void addTerrainPointLight(PointLights* pointLights, glm::vec3 pos);
// void addTerrainDirLight(Shader* shader, glm::vec3 pos);


void UserMenu() {
    // Convert currentTime to 24-hour format
    float hours = (currentTime / dayCycleDuration) * 24.0f;
    int intHours = static_cast<int>(hours);
    int hrs = (static_cast<int>(hours)+8)%24;
    int intMinutes = static_cast<int>((hours - intHours) * 60.0f);


    ImGui::Begin("Scene Info");  
    ImGui::Text("Current Time: %d:%d", hrs, intMinutes);
    ImGui::Text("Position (%.2f, %.2f, %.2f)", globCamera->getPosition().x, globCamera->getPosition().y, globCamera->getPosition().z);
    ImGui::SliderFloat("Height", &d0, globTerrain->minHeight, globTerrain->maxHeight, "%1.0f");
    ImGui::SameLine(); ImGui::Text("(%.2f, %.2f)", globTerrain->minHeight, globTerrain->maxHeight);
    if (ImGui::Button("Add Curve")) {
        curve_heights[curves] = d0;
        curves++;
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Curve")) {
        curves--;
    }
    ImGui::SameLine(); ImGui::Text("%d out of %d", curves, MAX_CURVES);
    if (ImGui::Button("Add Light")) {
        addTerrainPointLight(pointLightsSetup, globCamera->getPosition());
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Light")) {
        pointLightsSetup->removePointLight();
    }
    ImGui::SameLine(); ImGui::Text("%d out of %d", pointLightsSetup->nr_lights, NR_POINT_LIGHTS);

    if (useOrthoCamera) {
        if (ImGui::Button("Perspective camera")) {
            useOrthoCamera = !useOrthoCamera;
        }
    } else {
        if (ImGui::Button("Ortho camera")) {
            useOrthoCamera = !useOrthoCamera;
        }
    }


    ImGui::End();
}

void addTerrainPointLight(PointLights* pointLights, glm::vec3 pos) {
    int lts = pointLights->nr_lights;
    if (lts < NR_POINT_LIGHTS) {

        PointLight newPointLight(pos, defAmbient, defDiffuse, defSpecular, 
            defConstant, defLinear, defQuadratic);
        pointLights->addPointLight(newPointLight);

    }
}

// void addTerrainDirLight(Shader* shader, glm::vec3 pos) {
//     shader->setVec3("dirLight.direction", pos.x, pos.y, pos.z);
//     shader->setVec3("dirLight.ambient", defDirAmbient.x, defDirAmbient.y, defDirAmbient.z);
//     shader->setVec3("dirLight.diffuse", defDirDiffuse.x, defDirDiffuse.y, defDirDiffuse.z);
//     shader->setVec3("dirLight.specular", defDirSpecular.x, defDirSpecular.y, defDirSpecular.z);
// }

void updateScenelight(float deltaTime, Shader& shader) {
    currentTime += deltaTime;
    if (currentTime > dayCycleDuration) {
        currentTime -= dayCycleDuration; // Reset the time after a full cycle
    }

    // Calculate the angle of the sun (from 0 to 2 * PI)
    float angle = (currentTime / dayCycleDuration) * 2.0f * glm::pi<float>();

    // Sun direction - moving along the Z axis
    glm::vec3 sunDirection = glm::vec3(-cos(angle), 0.0f, -sin(angle));

    float hlfGrid = (float)GRID_SIZE/2.0f; 
    float hlfGridHeightAvg = (globTerrain->maxHeight + globTerrain->maxHeight)/2.0f;
    float hlfGridHeightDt = (globTerrain->maxHeight - globTerrain->maxHeight)/2.0f;
    // Moon position
    moonLight.setPosition(glm::vec3(-cos(angle)*hlfGrid + hlfGrid, hlfGrid, -sin(angle)*hlfGridHeightDt + hlfGridHeightAvg));

    // Calculate light intensity based on the angle
    float dayFactor = glm::clamp(static_cast<float>(sin(angle)), 0.0f, 1.0f);
    float nightFactor = 1.0f - dayFactor;

    
    moonLight.setAmbient(glm::vec3(0.05f, 0.05f, 0.1f) * nightFactor);
    moonLight.setDiffuse(glm::vec3(0.2f, 0.2f, 0.3f) * nightFactor);
    moonLight.setSpecular(glm::vec3(0.5f, 0.5f, 0.6f) * nightFactor);
    

    
    defDirDirection = sunDirection;
    shader.setVec3("dirLight.direction", sunDirection);
    shader.setVec3("dirLight.ambient", defDirAmbient * dayFactor);
    shader.setVec3("dirLight.diffuse", defDirDiffuse * dayFactor);
    shader.setVec3("dirLight.specular", defDirSpecular * dayFactor);
    std::string moonUniName = "moonLight";
    moonLight.setPointLight(&shader, moonUniName);
}

void updateClearColor(float dayFactor) {
    // Define colors for day and night
    glm::vec3 dayColor(0.53f, 0.81f, 0.98f);  // Light blue for day
    glm::vec3 nightColor(0.05f, 0.05f, 0.2f); // Dark blue for night

    // Interpolate between dayColor and nightColor based on dayFactor
    glm::vec3 clearColor = glm::mix(nightColor, dayColor, dayFactor);

    // Set the clear color
    glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
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
    glfwSetScrollCallback(window, scroll_callback);

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
    d0 = globTerrain->minHeight;
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

    Camera orthoCamera(SCR_WIDTH, SCR_HEIGHT);
    // orthoCamera.SetPosition((float)GRID_SIZE / 2.0f, (float)GRID_SIZE / 2.0f, globTerrain->maxHeight);
    orthoCamera.SetPosition(0.0f, 0.0f, globTerrain->maxHeight);
    orthoCamera.SetFront(0.0f, 0.0f, -1.0f);
    orthoCamera.SetUp(0.0f, 1.0f, 0.0f);

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

        float angle = (currentTime / dayCycleDuration) * 2.0f * glm::pi<float>();
        float dayFactor = glm::clamp(static_cast<float>(sin(angle)), 0.0f, 1.0f);
        updateClearColor(dayFactor);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        gui.newFrame();


        // activate shader
        ourShader.use();
        glm::vec3 lightPos((float)GRID_SIZE/2, (float)GRID_SIZE/2, terrain.maxHeight);

        // camera/view transformation
        // ourShader.setMat4("projection", camera.getProjection() );
        // ourShader.setMat4("view", camera.getView());
        ourShader.setMat4("model", camera.getModel());
        ourShader.setFloat("minHeight", terrain.minHeight);
        ourShader.setFloat("maxHeight", terrain.maxHeight);

        if (useOrthoCamera) {
            glm::mat4 orthoProjection = orthoCamera.getOrthographic(
                0.0f, (float)GRID_SIZE, 0.0f, (float)GRID_SIZE, 0.01f, globTerrain->maxHeight * 2.0f);
            ourShader.setMat4("projection", orthoProjection);
            ourShader.setMat4("view", orthoCamera.getView());
        } else {
            ourShader.setMat4("projection", globCamera->getProjection());
            ourShader.setMat4("view", globCamera->getView());
        }

        // illumination
        // addTerrainDirLight(&ourShader, glm::vec3(0.0f, 0.0f, -1.0f));
        updateScenelight(deltaTime, ourShader);
        pointLights.setPointLights();
        ourShader.setVec3("objectColor", objectColor.x, objectColor.y, objectColor.z);
        ourShader.setVec3("lightColor", lightColor.x, lightColor.y, lightColor.z);
        ourShader.setVec3("viewPos", camera.getPosition());

        GLTerrain.RenderTerrain(deltaTime*10.0f);

        
        geomShader.use();

        if (useOrthoCamera) {
            glm::mat4 orthoProjection = orthoCamera.getOrthographic(
                0.0f, (float)GRID_SIZE, 0.0f, (float)GRID_SIZE, 0.1f, globTerrain->maxHeight * 2.0f);
            geomShader.setMat4("projection", orthoProjection);
            geomShader.setMat4("view", orthoCamera.getView());
        } else {
            geomShader.setMat4("projection", globCamera->getProjection());
            geomShader.setMat4("view", globCamera->getView());
        }

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


        UserMenu();
        gui.drawData();

        
        if (!useOrthoCamera) camera.OnRender(deltaTime*10.0f);
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

    if (!useOrthoCamera) {
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
    }

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
        setup->reGenerateRandomTerrain();
        pointLightsSetup->clearPointLights();
        d0 = globTerrain->minHeight;
        curves = 0;    
    }
    
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    SCR_WIDTH = width; SCR_HEIGHT = height;
    globCamera->SetScrSize(width, height);
    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    globCamera->OnMouse((float)xposIn, (float)yposIn);
}

// // glfw: whenever the mouse scroll wheel scrolls, this callback is called
// // ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    globCamera->OnScroll(static_cast<float>(yoffset));
}