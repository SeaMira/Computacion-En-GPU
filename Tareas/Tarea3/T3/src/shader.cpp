#include <shader.h>
#include <glad/glad.h>

#define GL_CHECK_ERRORS(id, type, fun, log)                                    \
  ({                                                                           \
    int success;                                                               \
    char infoLog[512];                                                         \
    fun(id, type, &success);                                                   \
    if (!success) {                                                            \
      log(id, 512, NULL, infoLog);                                             \
      std::cerr << #type " (" #id ")\n" << infoLog << std::endl;               \
    }                                                                          \
  })

using std::string;

static inline void attachSource(unsigned int program, int type,
                                const string &path) {
  unsigned int shaderId = glCreateShader(type);
  std::fstream shaderFile;
  std::string shaderSource;
  shaderFile.exceptions(std::fstream::failbit | std::fstream::badbit);
  try {
    shaderFile.open(path);
    std::stringstream shaderBuffer;
    shaderBuffer << shaderFile.rdbuf();
    shaderFile.close();
    shaderSource = shaderBuffer.str();
    // std::cout << path << shaderSource << std::endl;
  } catch (std::fstream::failure e) {
    std::cerr << "File not succesfully read (" << path << ")" << std::endl;
  }
  const char *shaderSourcePtr = shaderSource.c_str();
  glShaderSource(shaderId, 1, &shaderSourcePtr, NULL);
  glCompileShader(shaderId);
  GL_CHECK_ERRORS(shaderId, GL_COMPILE_STATUS, glGetShaderiv,
                  glGetShaderInfoLog);
  glAttachShader(program, shaderId);
  glDeleteShader(shaderId);
}

Shader::Shader(const string &vertexPath, const string &fragmentPath) {
  std::cout << "create prog" << std::endl;
  id_ = glCreateProgram();
  attachSource(id_, GL_VERTEX_SHADER, vertexPath);
  attachSource(id_, GL_FRAGMENT_SHADER, fragmentPath);
  glLinkProgram(id_);
  GL_CHECK_ERRORS(id_, GL_LINK_STATUS, glGetProgramiv, glGetProgramInfoLog);
  std::cout << "Shaders cargados" << std::endl;
}

Shader::Shader(const string &vertexPath, const string &geometryPath,
               const string &fragmentPath) {
  id_ = glCreateProgram();
  attachSource(id_, GL_VERTEX_SHADER, vertexPath);
  attachSource(id_, GL_GEOMETRY_SHADER, geometryPath);
  attachSource(id_, GL_FRAGMENT_SHADER, fragmentPath);
  glLinkProgram(id_);
  GL_CHECK_ERRORS(id_, GL_LINK_STATUS, glGetProgramiv, glGetProgramInfoLog);
}

Shader::~Shader() { glDeleteProgram(id_); }

void Shader::use() { glUseProgram(id_); }

void Shader::set(const string &name, float value) {
  glUniform1f(get(name), value);
}

void Shader::set(const string &name, int value) {
  glUniform1i(get(name), value);
}

void Shader::set(const string &name, bool value) {
  glUniform1i(get(name), value);
}

void Shader::set(const string &name, const glm::mat4 &value) {
  glUniformMatrix4fv(get(name), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::set(const string &name, const glm::vec3 &value) {
  glUniform3fv(get(name), 1, glm::value_ptr(value));
}

int Shader::get(const string &name) {
  return glGetUniformLocation(id_, name.c_str());
}