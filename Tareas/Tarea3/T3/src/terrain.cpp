#include<terrain.h>

float noise = 50.0f;

// Función para inicializar la cuadrícula con valores aleatorios en los vértices de las esquinas
void initializeGrid(std::vector<std::vector<float>>& grid, int size, float roughness) {
    grid[0][0] = static_cast<float>(rand()) / RAND_MAX * roughness * noise - roughness;
    grid[0][size - 1] = static_cast<float>(rand()) / RAND_MAX * roughness * noise - roughness;
    grid[size - 1][0] = static_cast<float>(rand()) / RAND_MAX * roughness * noise - roughness;
    grid[size - 1][size - 1] = static_cast<float>(rand()) / RAND_MAX * roughness * noise - roughness;
}

// Función para realizar el paso "diamond" del algoritmo
void diamondStep(std::vector<std::vector<float>>& grid, int x, int y, int size, float offset) {
    int halfSize = size / 2;
    float avg = (grid[x][y] +
                 grid[x + size][y] +
                 grid[x][y + size] +
                 grid[x + size][y + size]) / 4.0f;
    grid[x + halfSize][y + halfSize] = avg + (static_cast<float>(rand()) / RAND_MAX * noise - 1.0f) * offset;
}

// Función para realizar el paso "square" del algoritmo
void squareStep(std::vector<std::vector<float>>& grid, int x, int y, int size, float offset) {
    int halfSize = size / 2;
    float avg = 0.0f;
    int count = 0;
    
    if (x >= halfSize) { // izquierda
        avg += grid[x - halfSize][y];
        count++;
    }
    if (x + halfSize < grid.size()) { // derecha
        avg += grid[x + halfSize][y];
        count++;
    }
    if (y >= halfSize) { // abajo
        avg += grid[x][y - halfSize];
        count++;
    }
    if (y + halfSize < grid.size()) { // arriba
        avg += grid[x][y + halfSize];
        count++;
    }

    avg /= count;
    grid[x][y] = avg + (static_cast<float>(rand()) / RAND_MAX * noise - 1.0f) * offset;
}

// Función para generar el terreno utilizando el algoritmo Diamond-Square
void generateFractalTerrain(std::vector<std::vector<float>>& grid, int size, float roughness) {
    int stepSize = size - 1;
    float offset = roughness;

    while (stepSize > 1) {
        for (int x = 0; x < size - 1; x += stepSize) {
            for (int y = 0; y < size - 1; y += stepSize) {
                diamondStep(grid, x, y, stepSize, offset);
            }
        }

        for (int x = 0; x < size; x += stepSize / 2) {
            for (int y = (x + stepSize / 2) % stepSize; y < size; y += stepSize) {
                squareStep(grid, x, y, stepSize, offset);
            }
        }

        stepSize /= 2;
        offset *= std::pow(2, -roughness);
    }
}

// Función para generar un terreno aleatorio
void Terrain::generateRandomTerrain(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo para escribir." << std::endl;
        return;
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::vector<std::vector<float>> grid(gridSize, std::vector<float>(gridSize));
    std::cout << "initialize vector grid" << std::endl;
    initializeGrid(grid, gridSize, roughness);
    std::cout << "initialize grid" << std::endl;
    generateFractalTerrain(grid, gridSize, roughness);
    std::cout << "generate fractal terrain" << std::endl;

    float minHeight = 20.0f;
    float maxHeight = 0.0f;

    // Generar vértices y determinar los valores mínimo y máximo de z
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            float z = grid[x][y];
            if (z < minHeight) minHeight = z;
            if (z > maxHeight) maxHeight = z;
            vertices.push_back(Vertex((float)x, (float)y, z, 0.0f, 0.0f, 0.0f)); // Colores inicializados en 0
        }
    }

    // Normalizar los colores según la altura
    for (auto& vertex : vertices) {
        float normalizedZ = (vertex.z - minHeight) / (maxHeight - minHeight);
        vertex.r = vertex.g = vertex.b = normalizedZ;
    }

    // Generar triángulos
    for (unsigned int x = 0; x < gridSize - 1; ++x) {
        for (unsigned int y = 0; y < gridSize - 1; ++y) {
            unsigned int currentIndex = x * gridSize + y;
            unsigned int nextRowIndex = (x + 1) * gridSize + y;

            // Triángulo 1
            indices.push_back(Triangle(currentIndex, nextRowIndex, currentIndex + 1));

            // Triángulo 2
            indices.push_back(Triangle(nextRowIndex, nextRowIndex + 1, currentIndex + 1));
        }
    }

    // Escribir vértices en el archivo
    for (const auto& vertex : vertices) {
        file << "V " << vertex.x << " " << vertex.y << " " << vertex.z << " " << vertex.r << " " << vertex.g << " " << vertex.b << std::endl;
    }

    // Escribir triángulos en el archivo
    for (const auto& triangle : indices) {
        file << "I " << triangle.v1 << " " << triangle.v2 << " " << triangle.v3 << std::endl;
    }

    file.close();
    std::cout << "Terreno fractal generado y guardado en " << filename << std::endl;
}


void Terrain::loadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        char type;
        iss >> type;

        if (type == 'V') {
            float x, y, z, r, g, b;
            iss >> x >> y >> z >> r >> g >> b;
            vertices.push_back(Vertex(x, y, z, r, g, b));
        }
        else if (type == 'I') {
            unsigned int v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            indices.push_back(Triangle(v1, v2, v3)) ;
        }
    }

    file.close();
}

Vertex* Terrain::getVerticesData() {
    return &vertices[0];
}

int Terrain::verticesSize() {
    return vertices.size();
}

Triangle* Terrain::getTrianglesData() {
    return &indices[0];
}

int Terrain::trianglesSize() {
    return indices.size();
}