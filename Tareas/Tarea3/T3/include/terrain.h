#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

// Estructura para un vértice
struct Vertex {
    float x, y, z;
    Vertex(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Estructura para un triángulo
struct Triangle {
    int v1, v2, v3;
    Triangle(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}
};

// Estructura para un color
struct Color {
    int i;
    float r, g, b;
    Color(int i, float r, float g, float b) : i(i), r(r), g(g), b(b) {}
};

class Terrain {
    private:
        int width, height;
        std::vector<Vertex> vertices;
        std::vector<Triangle> indices;
        std::vector<Color> colores;
    public:
        Terrain(int width, int height) : width(width), height(height) {}
        void generateRandomTerrain(const std::string& filename);
        void loadFromFile(const std::string& filename);
        void draw();
};