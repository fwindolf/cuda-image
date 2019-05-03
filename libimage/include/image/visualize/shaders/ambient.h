#pragma once
#include <string>

namespace pangolin {

const std::string ambient_light_shader = R"Shader(
/////////////////////////////////////////
@start vertex
#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec4 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 FragPos;
out vec4 Normal;

void main()
{
    gl_Position = projection * view * model * aPos;
    FragPos = model * aPos;
    Normal = transpose(inverse(model)) * aNormal;
} 

/////////////////////////////////////////
@start fragment
#version 330 core

in vec4 FragPos;
in vec4 Normal;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 ambient;

out vec4 FragColor;

void main()
{
    vec4 norm = normalize(Normal);
    vec4 lightDir = normalize(vec4(lightPos, 1.0) - FragPos);
    
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;

    FragColor = vec4(result, 1.0);
    
}
)Shader";

}