#include<terrain.h>

float noise(float x, float y) {
    return (std::sin(x * 0.1f) * std::cos(y * 0.1f)) * 20.0f + (std::rand() % 100);
}

// Función para generar un terreno aleatorio
void Terrain::generateRandomTerrain(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo para escribir." << std::endl;
        return;
    }

    // Generar vértices
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            float z = noise(static_cast<float>(x), static_cast<float>(y));
            z = std::fmod(z, 6000.0f);
            if (z < 0.0f) z += 6000.0f;
            vertices.push_back(Vertex(x, y, z, std::rand() / static_cast<float>(RAND_MAX), 
                                std::rand() / static_cast<float>(RAND_MAX), 
                                std::rand() / static_cast<float>(RAND_MAX)));

        }
    }

    // Generar triángulos
    for (unsigned int x = 0; x < width - 1; ++x) {
        for (unsigned int y = 0; y < height - 1; ++y) {
            unsigned int currentIndex = x * height + y;
            unsigned int nextRowIndex = (x + 1) * height + y;

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
    std::cout << "Terreno aleatorio generado y guardado en " << filename << std::endl;
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
    return vertices.data();
}

int Terrain::verticesSize() {
    return vertices.size();
}

Triangle* Terrain::getTrianglesData() {
    return indices.data();
}

int Terrain::trianglesSize() {
    return indices.size();
}