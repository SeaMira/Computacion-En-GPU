#include <glad/glad.h>
#include <interface.h>
#include <camera3.h>
#include <shader_m.h>
#include <terrain.h>
#include <iostream>


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
        GLuint terrainNbo = -1;
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

            glGenBuffers(1, &terrainNbo);
            glBindBuffer(GL_ARRAY_BUFFER, terrainNbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Normal)*terrain->normalesSize(), terrain->getNormalesData(), GL_STATIC_DRAW);
           
            
        }

        void reCreateTerrainArraysBuffers() {
            glBindVertexArray(terrainVao);

            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*terrain->verticesSize(), terrain->getVerticesData(), GL_STATIC_DRAW);

            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle)*terrain->trianglesSize(), terrain->getTrianglesData(), GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, terrainNbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Normal)*terrain->normalesSize(), terrain->getNormalesData(), GL_STATIC_DRAW);
           
            
        }

        void RenderTerrain(float dt) {
            
            
            glBindVertexArray(terrainVao);

            glBindBuffer(GL_ARRAY_BUFFER, terrainVbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIbo);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);

            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));

            glBindBuffer(GL_ARRAY_BUFFER, terrainNbo);
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Normal), (void*)0);

            glDrawElements(GL_TRIANGLES, (terrain->trianglesSize())*3, GL_UNSIGNED_INT, 0);
            
            // glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glBindVertexArray(0);
            glDisableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            // std::cout << "Rendered frame." << std::endl;
            
        }

        void reGenerateRandomTerrain() {
            terrain->reGenerateRandomTerrain();
            reCreateTerrainArraysBuffers();
        }

};