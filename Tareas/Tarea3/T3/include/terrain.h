#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cmath>
#include <limits>



// Estructura para un vértice
struct Vertex {
    float x, y, z;
    float r, g, b;
    Vertex(float x, float y, float z, float r, float g, float b) : x(x), y(y), z(z), r(r), g(g), b(b) {}
};

// Estructura para un triángulo
struct Triangle {
    unsigned int v1, v2, v3;
    Triangle(unsigned int v1, unsigned int v2, unsigned int v3) : v1(v1), v2(v2), v3(v3) {}
};

class Normal {
    private:
        float u, v, w;
    public:
        Normal(float u, float v, float w) : u(u), v(v), w(w) {}
        float getU() {return u;}
        float getV() {return v;}
        float getW() {return w;}

        Normal operator+(Normal& n) {
            return Normal(this->u + n.getU(),
                this->v + n.getV(),
                this->w + n.getW());
        } 

        void operator+=(Normal& n) {
            this->u += n.getU();
            this->v += n.getV();
            this->w += n.getW();
        } 

        void normalize() {
            float norm = sqrt(u*u+v*v+w*w);
            if (norm != 0.0f) {
                u /= norm;
                v /= norm;
                w /= norm;
            }
        }
};



class Terrain {
    private:
        int gridSize;
        float roughness;
        
        std::vector<Vertex> vertices;
        std::vector<Triangle> indices;
        std::vector<Normal> normales;

        std::string filename;
    public:
        Terrain(int gridSize, float roughness) : gridSize(gridSize),  roughness(roughness) {
            std::srand(static_cast<unsigned int>(std::time(0)));
        }
        void generateRandomTerrain(const std::string& filename);
        void reGenerateRandomTerrain();
        void loadFromFile(const std::string& filename);
        Vertex* getVerticesData();
        int verticesSize();
        Triangle* getTrianglesData();
        int trianglesSize();
        Normal* getNormalesData();
        int normalesSize();
        
        float minHeight, maxHeight;
};