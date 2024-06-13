#include<iostream>
#include"terrain.h"
#include"shader.h"
#include"camera.h"
#include"init.h"

int height 100, width = 100;


const std::string vertex_shader_path = "vertexShader.txt"
const std::string fragment_shader_path = "fragmentShader.txt"
const std::string terrain_path = "terrain.txt"

int main(int argc, char const *argv[]) {

    Shader shaderObj(vertex_shader_path, fragment_shader_path);
    shaderObj.use();
   
   Terrain terrain(width, height);
   terrain.generateRandomTerrain(terrain_path);
   terrain.loadFromFile(terrain_path);
}

