#version 430

layout(std140, set = 0, binding = 0) uniform Args {
    vec3 light_position;
};
layout(set = 1, binding = 0) uniform sampler fullscreen_sampler;
layout(set = 1, binding = 1) uniform texture2D texture_position;


layout(location = 0) in vec3 frag_center;
layout(location = 1) in float frag_radius;

layout(location = 0) out float shadow;

float is_shadow(const float t) {
    return t > 0.0 && t < 0.99 ? 0.0 : 1.0;
}

void main() {
    vec3 position = texture(sampler2D(texture_position, fullscreen_sampler), gl_FragCoord.xy).xyz;

    vec3 direction = position - light_position;
    vec3 oc = light_position - frag_center;

    float a = dot(direction, direction);
    float b = 2.0 * dot(oc, direction);
    float c = dot(oc,oc) - frag_radius*frag_radius;
    float discriminant = b*b - 4.0*a*c;

    if(discriminant > 0.0){
        float t = (-b - sqrt(discriminant)) / (2.0*a);
        shadow = is_shadow(t);
    } else if (discriminant == 0.0) {
        float t = (-b) / (2.0*a);
        shadow = is_shadow(t);
    } else {
        shadow = 1.0;
    }
}