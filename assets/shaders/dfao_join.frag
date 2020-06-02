#version 430

layout(std140, set = 0, binding = 0) uniform Args {
    float offset;
    float factor;
};

layout(set = 1, binding = 0) uniform sampler fullscreen_sampler;
layout(set = 1, binding = 1) uniform texture2D texture_distance;

layout(location = 0) out vec4 out_occlusion;

void main() {
    float distance = texture(sampler2D(texture_distance, fullscreen_sampler), gl_FragCoord.xy).x;

    out_occlusion = vec4((offset - distance) * factor, 0.0, 0.0, 0.0);
    //out_occlusion = vec4(distance * 0.01, 0.0, 0.0, 0.0);
}
