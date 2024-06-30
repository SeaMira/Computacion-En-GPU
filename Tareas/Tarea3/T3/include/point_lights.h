#ifndef _POINT_LIGHTS_H
#define _POINT_LIGHTS_H

#include "shader_m.h"
#include <vector>

#define NR_POINT_LIGHTS 80


class PointLight {
    public: 
        PointLight(glm::vec3 position, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular,float constant, float linear, float quadratic) : 
            position(position), ambient(ambient), diffuse(diffuse), specular(specular), constant(constant), linear(linear), quadratic(quadratic) {}

        void setPointLight(Shader* sh, int i);
        void setPointLight(Shader* sh, const std::string& uniName);

        void setPosition(glm::vec3 position);
        void setAmbient(glm::vec3 ambient);
        void setDiffuse(glm::vec3 diffuse);
        void setSpecular(glm::vec3 specular);

        void setConstant(float constant);
        void setLinear(float linear);
        void setQuadratic(float quadratic);

        glm::vec3 getPosition();
        glm::vec3 getAmbient();
        glm::vec3 getDiffuse();
        glm::vec3 getSpecular();

        float getConstant();
        float getLinear();
        float getQuadratic();

    private:
        glm::vec3 position;
    
        glm::vec3 ambient;
        glm::vec3 diffuse;
        glm::vec3 specular;

        float constant;
        float linear;
        float quadratic;     
};


class PointLights {
    public:
        PointLights(Shader* shader): shader(shader) {}
        PointLights(Shader* shader, std::vector<PointLight> lights): 
            shader(shader), lights(lights) {
                nr_lights = lights.size();
            }

        void addPointLight(PointLight pointLight);
        void removePointLight();
        void setPointLights();
        void clearPointLights();

        int nr_lights = 0;
    private:
        std::vector<PointLight> lights;
        Shader* shader;
};

#endif // _POINT_LIGHTS_H