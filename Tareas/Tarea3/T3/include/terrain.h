#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cmath>

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


class Terrain {
    private:
        int width, height;
        std::vector<Vertex> vertices;
        std::vector<Triangle> indices;
    public:
        Terrain(int width, int height) : width(width), height(height) {}
        void generateRandomTerrain(const std::string& filename);
        void loadFromFile(const std::string& filename);
        Vertex* getVerticesData();
        int verticesSize();
        Triangle* getTrianglesData();
        int trianglesSize();
};