#pragma once

#ifndef _SHADER_H_
#define _SHADER_H_

#include <string>
#include <glad/glad.h>
#include "glm/gtc/type_ptr.hpp"
#include "glm/fwd.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
class Shader {
public:
  Shader(const std::string &vertexPath, const std::string &fragmentPath);
  Shader(const std::string &vertexPath, const std::string &geometryPath,
         const std::string &fragmentPath);
  Shader(Shader &&) = default;
  Shader(const Shader &) = default;
  Shader &operator=(Shader &&) = default;
  Shader &operator=(const Shader &) = default;
  ~Shader();

  void use();
  void set(const std::string &name, float value);
  void set(const std::string &name, int value);
  void set(const std::string &name, bool value);
  void set(const std::string &name, const glm::mat4 &value);
  void set(const std::string &name, const glm::vec3 &value);
  int get(const std::string &name);

private:
  unsigned int id_;
};

#endif // _SHADER_H_