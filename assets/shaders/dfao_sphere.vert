#version 430

layout(std140, set = 0, binding = 0) uniform Args {
    mat4 projection_matrix;
    float offset;
};

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 sphere_center;
layout(location = 2) in float sphere_radius;

layout(location = 0) out vec3 frag_center;
layout(location = 1) out float frag_radius;

void main() {
    frag_center = sphere_center;
    frag_radius = sphere_radius;

    gl_Position = projection_matrix * vec4(a_pos * (sphere_radius + offset) + sphere_center, 1.0);
}
