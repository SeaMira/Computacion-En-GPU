#include<terrain.h>

float noise(float x, float y) {
    return (std::sin(x * 0.1f) * std::cos(y * 0.1f)) * 20.0f + (std::rand() % 5);
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
            vertices.emplace_back(x, y, z);
            colores.emplace_back(x * height + y, std::rand() / static_cast<float>(RAND_MAX), 
                                std::rand() / static_cast<float>(RAND_MAX), 
                                std::rand() / static_cast<float>(RAND_MAX));
        }
    }

    // Generar triángulos
    for (int x = 0; x < width - 1; ++x) {
        for (int y = 0; y < height - 1; ++y) {
            int currentIndex = x * height + y;
            int nextRowIndex = (x + 1) * height + y;

            // Triángulo 1
            indices.emplace_back(currentIndex, nextRowIndex, currentIndex + 1);

            // Triángulo 2
            indices.emplace_back(nextRowIndex, nextRowIndex + 1, currentIndex + 1);
        }
    }

    // Escribir vértices en el archivo
    for (const auto& vertex : vertices) {
        file << "V " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
    }

    // Escribir triángulos en el archivo
    for (const auto& triangle : indices) {
        file << "I " << triangle.v1 << " " << triangle.v2 << " " << triangle.v3 << std::endl;
    }

    // Escribir colores en el archivo
    for (const auto& color : colores) {
        file << "T " << color.i << " " << color.r << " " << color.g << " " << color.b << std::endl;
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
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(Vertex(x, y, z));
        }
        else if (type == 'I') {
            unsigned int v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            indices.push_back(Triangle(v1, v2, v3)) ;
        }
        else if (type == 'T') {
            int i;
            float r, g, b;
            iss >> i >> r >> g >> b;
            if (i >= 0 && i < vertices.size()) {
                colores.push_back(Color(i, r, g, b));
            } else {
                std::cerr << "Índice de color fuera de rango: " << i << std::endl;
            }
        }
    }

    file.close();

    // Ensure that the colors vector is the same size as the vertices vector
    if (colores.size() != vertices.size()) {
        std::cerr << "Número de colores no coincide con el número de vértices" << std::endl;
        int i = colores.size();
        while (colores.size() < vertices.size()) {
            colores.push_back(Color(i, 1.0f, 1.0f, 1.0f)); // Añadir color blanco por defecto
            i++;
        }
    }
}