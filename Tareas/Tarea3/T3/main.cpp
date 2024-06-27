#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <camera3.h>
#include <shader_m.h>
#include <terrain.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
// void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window, float deltaTime);

// settings
unsigned int SCR_WIDTH;
unsigned int SCR_HEIGHT;
int GRID_SIZE;

// camera
Camera* globCamera;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;


class TerrainSetup {
    private:
        Terrain* terrain;
        // Camera* camera;
        // WorldTrans* wrldTrans;
        // ProjectionTrans* projInfo;
        // glm::mat4 mat;

        // GLuint gWVPLocation;

        GLuint terrainVao = -1;
        GLuint terrainVbo = -1;
        GLuint terrainIbo = -1;

        int LOCAL_SIZE, GROUP_SIZE;

    public:
        TerrainSetup(Terrain* terrain) {
            this->terrain = terrain;
            // this->camera = camera;
            // this->wrldTrans = wrldTrans;
            // this->projInfo = projInfo;

            // mat = (projInfo->getProjectionMatrix()) *(camera->GetMatrix()) * (wrldTrans->GetMatrix());
        }

        

        ~TerrainSetup() {
            if (terrain) {
                delete terrain;
            }
            // if (camera) {
            //     delete camera;
            // }
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

            
            glGenBuffers(1, &terrainIbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle)*terrain->trianglesSize(), terrain->getTrianglesData(), GL_STATIC_DRAW);
           
            
        }

        void RenderTerrain(float dt) {
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            glBindVertexArray(terrainVao);

            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);

            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
            glDrawElements(GL_TRIANGLES, (terrain->trianglesSize())*3, GL_UNSIGNED_INT, 0);
            
            // glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            // std::cout << "Rendered frame." << std::endl;
            
        }

};






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

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("vertexShader.txt", "fragmentShader.txt");


    Terrain terrain(GRID_SIZE, roughness);
    terrain.generateRandomTerrain("terrain.txt");
    TerrainSetup GLTerrain(&terrain);
    GLTerrain.CreateTerrainVAO();    

    glLineWidth(1); 
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    // -------------------------------------------------------------------------------------------
    ourShader.use();
    
    Camera camera(SCR_WIDTH, SCR_HEIGHT);
    globCamera = &camera;

    // pass projection matrix to shader (as projection matrix rarely changes there's no need to do this per frame)
    // -----------------------------------------------------------------------------------------------------------
    ourShader.setMat4("projection", camera.getProjection() );


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

        // camera/view transformation
        ourShader.setMat4("view", camera.getView());

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
        ourShader.setMat4("model", camera.getModel());
        GLTerrain.RenderTerrain(deltaTime*10.0f);
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