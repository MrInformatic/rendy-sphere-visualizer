#version 450

layout(set = 0, binding = 0) uniform samplerCube environment;

layout(std140, set = 1, binding = 0) uniform Args {
    mat4 inversed_view_matrix;
    vec3 ambient;
    vec3 light_color;
    vec3 light_position;
};
layout(set = 1, binding = 1) uniform sampler fullscreen_sampler;
layout(set = 1, binding = 2) uniform texture2D texture_position;
layout(set = 1, binding = 3) uniform texture2D texture_normal;
layout(set = 1, binding = 4) uniform texture2D texture_color;
layout(set = 1, binding = 5) uniform texture2D texture_n;
layout(set = 1, binding = 6) uniform texture2D texture_occlusion;
layout(set = 1, binding = 7) uniform texture2D texture_shadow;

layout(location = 0) out vec4 o_color;

void main() {
    vec2 frag_coord = gl_FragCoord.xy;

    vec3 position = texture(sampler2D(texture_position, fullscreen_sampler), frag_coord).xyz;
    vec3 normal =  texture(sampler2D(texture_normal, fullscreen_sampler), frag_coord).xyz;
    vec3 color = texture(sampler2D(texture_color, fullscreen_sampler), frag_coord).rgb;
    float n = texture(sampler2D(texture_n, fullscreen_sampler), frag_coord).r;
    float occlusion = texture(sampler2D(texture_occlusion, fullscreen_sampler), frag_coord).r;
    float shadow = texture(sampler2D(texture_shadow, fullscreen_sampler), frag_coord).r;

    float normal_length = dot(normal, normal);

    if(normal_length > 0.1) {
        normal = normal / inversesqrt(normal_length);
        vec3 camera_dir = normalize(position);

        vec3 light_dir = light_position - position;
        float squared_length_light_dir = dot(light_dir, light_dir);
        vec3 normalized_light_dir = light_dir * inversesqrt(squared_length_light_dir);
        float light_dot = dot(normalized_light_dir, normal);
        vec3 diffuse = light_color / squared_length_light_dir * max(light_dot, 0.0);

        float dot = dot(camera_dir, normal);
        float r = (1.0 - n) / (1.0 + n);
        float r2 = r*r;
        float shlick = r2 + (1.0 - r2) * pow(1.0 + dot, 5.0);
        vec3 reflection_dir = (inversed_view_matrix * vec4(camera_dir + normal * (-2.0 * dot), 0)).xyz;

        vec3 reflection = texture(environment, reflection_dir).xyz;

        o_color = vec4(mix((ambient * occlusion + diffuse * shadow) * color, reflection * occlusion * (light_dot > 0.1 ? shadow : 1.0), clamp(shlick, 0.0, 1.0)), 1.0);
    } else {
        o_color = vec4(texture(environment, vec3(0, 0, 1)).xyz, 1.0);
    }
}