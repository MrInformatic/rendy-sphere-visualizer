#version 430

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_norm;
layout(location = 2) in vec3 frag_color;
layout(location = 3) in float frag_n;

layout(location = 0) out vec4 out_pos;
layout(location = 1) out vec4 out_norm;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec4 out_n;

void main() {
    out_pos = vec4(frag_pos, 1.0);
    out_norm = vec4(normalize(frag_norm), 1.0);
    out_color = vec4(frag_color, 1.0);
    out_n = vec4(frag_n, 1.0, 1.0, 1.0);
}
