#version 430

layout(std140, set = 0, binding = 0) uniform Args {
    mat4 projection_matrix;
    float offset;
};

layout(set = 1, binding = 0) uniform sampler fullscreen_sampler;
layout(set = 1, binding = 1) uniform texture2D texture_position;
layout(set = 1, binding = 2) uniform texture2D texture_normal;

layout(location = 0) in vec3 frag_center;
layout(location = 1) in float frag_radius;

layout(location = 0) out vec4 out_distance;

vec3 normal_from_unorm(vec3 normal) {
    return vec3(
        (normal.x - 0.5) * 2.0,
        (normal.y - 0.5) * 2.0,
        normal.z
    );
}

void main() {
    vec3 pos = texture(sampler2D(texture_position, fullscreen_sampler), gl_FragCoord.xy).xyz;
    vec3 norm = normal_from_unorm(texture(sampler2D(texture_normal, fullscreen_sampler), gl_FragCoord.xy).xyz);

    out_distance = vec4(distance(pos + norm * offset, frag_center) - frag_radius, 0.0, 0.0, 0.0);
}
