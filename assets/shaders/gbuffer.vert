#version 430

layout(std140, binding = 0) uniform Args {
    mat4 proj;
};

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_norm;

// per-instance
layout(location = 2) in vec4 model_view[4]; // vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
layout(location = 6) in vec3 model_view_norm[3]; // vec3[3] is used instead of mat3 due to spirv-cross bug for dx12 backend

// per-instance
layout(location = 9) in vec3 mat_color;
layout(location = 10) in float mat_n;

layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec3 frag_norm;
layout(location = 2) out vec3 frag_color;
layout(location = 3) out float frag_n;

void main() {
    mat4 model_view_mat = mat4(model_view[0], model_view[1], model_view[2], model_view[3]);
    mat3 model_view_norm_mat = mat3(model_view_norm[0], model_view_norm[1], model_view_norm[2]);

    frag_norm = model_view_norm_mat * a_norm;
    vec4 pos = model_view_mat * vec4(a_pos, 1.0);
    frag_pos = pos.xyz;
    frag_n = mat_n;
    frag_color = mat_color;
    gl_Position = proj * pos;
}