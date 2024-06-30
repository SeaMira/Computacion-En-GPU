#include <point_lights.h>


void PointLight::setPointLight(Shader* sh, int i) {
    std::stringstream pos, am, diff, spec, cons, lin, quad;
    pos << "pointLights[" << i << "].position";
    sh->setVec3(pos.str().c_str(), position.x, position.y, position.z);

    am << "pointLights[" << i << "].ambient";
    sh->setVec3(am.str().c_str(), ambient.x, ambient.y, ambient.z);

    diff << "pointLights[" << i << "].diffuse";
    sh->setVec3(diff.str().c_str(), diffuse.x, diffuse.y, diffuse.z);

    spec << "pointLights[" << i << "].specular";
    sh->setVec3(spec.str().c_str(), specular.x, specular.y, specular.z);

    cons << "pointLights[" << i << "].constant";
    sh->setFloat(cons.str().c_str(), constant);

    lin << "pointLights[" << i << "].linear";
    sh->setFloat(lin.str().c_str(), linear);

    quad << "pointLights[" << i << "].quadratic";
    sh->setFloat(quad.str().c_str(), quadratic);

}

void PointLight::setPointLight(Shader* sh, const std::string& uniName) {
    std::stringstream pos, am, diff, spec, cons, lin, quad;
    pos << uniName << ".position";
    sh->setVec3(pos.str().c_str(), position.x, position.y, position.z);

    am << uniName << ".ambient";
    sh->setVec3(am.str().c_str(), ambient.x, ambient.y, ambient.z);

    diff << uniName << ".diffuse";
    sh->setVec3(diff.str().c_str(), diffuse.x, diffuse.y, diffuse.z);

    spec << uniName << ".specular";
    sh->setVec3(spec.str().c_str(), specular.x, specular.y, specular.z);

    cons << uniName << ".constant";
    sh->setFloat(cons.str().c_str(), constant);

    lin << uniName << ".linear";
    sh->setFloat(lin.str().c_str(), linear);

    quad << uniName << ".quadratic";
    sh->setFloat(quad.str().c_str(), quadratic);

}

void PointLight::setPosition(glm::vec3 position) {
    this->position = position;
}

void PointLight::setAmbient(glm::vec3 ambient) {
    this->ambient = ambient;
}

void PointLight::setDiffuse(glm::vec3 diffuse) {
    this->diffuse = diffuse;
}
void PointLight::setSpecular(glm::vec3 specular) {
    this->specular = specular;
}
void PointLight::setConstant(float constant) {
    this->constant = constant;
}
void PointLight::setLinear(float linear) {
    this->linear = linear;
}
void PointLight::setQuadratic(float quadratic) {
    this->quadratic = quadratic;
}

glm::vec3 PointLight::getPosition() {
    return position;
}
glm::vec3 PointLight::getAmbient() {
    return ambient;
}
glm::vec3 PointLight::getDiffuse() {
    return diffuse;
}
glm::vec3 PointLight::getSpecular() {
    return specular;
}

float PointLight::getConstant() {
    return constant;
}
float PointLight::getLinear() {
    return linear;
}
float PointLight::getQuadratic() {
    return quadratic;
}



void PointLights::addPointLight(PointLight pointLight) {
    std::cout << "addPointLight " << nr_lights << std::endl;
    if (nr_lights < NR_POINT_LIGHTS) {
        lights.push_back(pointLight);
        nr_lights++;
    }
}

void PointLights::setPointLights() {
    shader->setInt("point_lights", nr_lights);
    for (int i = 0; i < nr_lights; i++) {
        lights[i].setPointLight(shader, i);
    }
}

void PointLights::removePointLight() {
    std::cout << "removePointLight " << nr_lights << std::endl;
    if (nr_lights > 0) {
        lights.pop_back();
        nr_lights--;
    }
}


void PointLights::clearPointLights() {
    lights.clear();
    nr_lights = 0;
}
