#version 430

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4[4] model_view_projection; // vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
layout(location = 5) in vec3 sphere_center;
layout(location = 6) in float sphere_radius;

layout(location = 0) out vec3 frag_center;
layout(location = 1) out float frag_radius;

void main() {
    mat4 mat_model_view_projection = mat4(model_view_projection[0], model_view_projection[1], model_view_projection[2], model_view_projection[3]);

    frag_center = sphere_center;
    frag_radius = sphere_radius;

    gl_Position = mat_model_view_projection * vec4(a_position, 1.0);
}
