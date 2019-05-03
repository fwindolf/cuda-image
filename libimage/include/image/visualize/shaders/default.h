#pragma once
#include <string>

namespace pangolin {

const std::string default_model_shader = R"Shader(
/////////////////////////////////////////
@start vertex
#version 330 core

layout(location = 0)in vec4 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vec4 vertexPos = projection * view * model * aPos;    
    gl_Position = vertexPos;
}

/////////////////////////////////////////
@start fragment
#version 330 core

uniform vec3 objectColor;
uniform vec3 lightColor;

out vec4 FragColor;

void main()
{
    vec3 result = lightColor * objectColor;
    FragColor = vec4(result, 1.0);
}
)Shader";



}