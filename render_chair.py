import trimesh
import numpy as np
import glfw
import glm
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import sys
import ctypes

# ============ 加载 GLB 文件 ============

scene = trimesh.load("/home/ai/Desktop/chair.glb")

cam = scene.camera
cam_transform = scene.camera_transform
cam_pos = cam_transform[:3, 3].copy()
cam_fov = float(cam.fov[1])
cam_res = cam.resolution

mesh_data = []
for name, geom in scene.geometry.items():
    mat = geom.visual.material
    metallic = mat.metallicFactor if mat.metallicFactor is not None else 0.0
    roughness = 0.7
    mesh_data.append({
        "vertices": np.array(geom.vertices, dtype=np.float32),
        "normals": np.array(geom.vertex_normals, dtype=np.float32),
        "uvs": np.array(geom.visual.uv, dtype=np.float32),
        "faces": np.array(geom.faces, dtype=np.uint32),
        "tex_img": mat.baseColorTexture,
        "metallic": metallic,
        "roughness": roughness,
    })

combined = trimesh.util.concatenate(scene.dump())
bounds = combined.bounds
center = (bounds[0] + bounds[1]) / 2.0

# ============ 加载 cubemap ============

def load_cubemap_faces(folder):
    names = {"px": "posx.jpg", "nx": "negx.jpg", "py": "posy.jpg",
             "ny": "negy.jpg", "pz": "posz.jpg", "nz": "negz.jpg"}
    faces = {}
    for key, filename in names.items():
        faces[key] = Image.open(f"{folder}/{filename}").convert("RGB")
    return faces, faces["px"].width

# ============ GLSL 着色器 ============

CUBE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec3 localPos;
uniform mat4 projection;
uniform mat4 view;
void main(){
    localPos = aPos;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
"""

IRRADIANCE_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube environmentMap;
const float PI = 3.14159265359;
void main(){
    vec3 normal = normalize(localPos);
    vec3 irradiance = vec3(0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));
    float sampleDelta = 0.025;
    float nrSamples = 0.0;
    for(float phi = 0.0; phi < 2.0*PI; phi += sampleDelta){
        for(float theta = 0.0; theta < 0.5*PI; theta += sampleDelta){
            vec3 tangentSample = vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
            vec3 sampleVec = tangentSample.x*right + tangentSample.y*up + tangentSample.z*normal;
            irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance / nrSamples;
    FragColor = vec4(irradiance, 1.0);
}
"""

PREFILTER_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube environmentMap;
uniform float roughness;
const float PI = 3.14159265359;
float RadicalInverse_VdC(uint bits){
    bits = (bits<<16u)|(bits>>16u);
    bits = ((bits&0x55555555u)<<1u)|((bits&0xAAAAAAAAu)>>1u);
    bits = ((bits&0x33333333u)<<2u)|((bits&0xCCCCCCCCu)>>2u);
    bits = ((bits&0x0F0F0F0Fu)<<4u)|((bits&0xF0F0F0F0u)>>4u);
    bits = ((bits&0x00FF00FFu)<<8u)|((bits&0xFF00FF00u)>>8u);
    return float(bits)*2.3283064365386963e-10;
}
vec2 Hammersley(uint i, uint N){ return vec2(float(i)/float(N), RadicalInverse_VdC(i)); }
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float r){
    float a=r*r; float phi=2.0*PI*Xi.x;
    float cosTheta=sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
    float sinTheta=sqrt(1.0-cosTheta*cosTheta);
    vec3 H=vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
    vec3 up=abs(N.z)<0.999?vec3(0,0,1):vec3(1,0,0);
    vec3 tangent=normalize(cross(up,N));
    vec3 bitangent=cross(N,tangent);
    return normalize(tangent*H.x+bitangent*H.y+N*H.z);
}
void main(){
    vec3 N=normalize(localPos); vec3 R=N; vec3 V=R;
    const uint SAMPLE_COUNT=1024u;
    float totalWeight=0.0; vec3 prefilteredColor=vec3(0.0);
    for(uint i=0u;i<SAMPLE_COUNT;i++){
        vec2 Xi=Hammersley(i,SAMPLE_COUNT);
        vec3 H=ImportanceSampleGGX(Xi,N,roughness);
        vec3 L=normalize(2.0*dot(V,H)*H-V);
        float NdotL=max(dot(N,L),0.0);
        if(NdotL>0.0){ prefilteredColor+=texture(environmentMap,L).rgb*NdotL; totalWeight+=NdotL; }
    }
    prefilteredColor/=totalWeight;
    FragColor=vec4(prefilteredColor,1.0);
}
"""

BRDF_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
out vec2 TexCoords;
void main(){ TexCoords=aUV; gl_Position=vec4(aPos,1.0); }
"""

BRDF_FS = """
#version 330 core
out vec2 FragColor;
in vec2 TexCoords;
const float PI = 3.14159265359;
float RadicalInverse_VdC(uint bits){
    bits=(bits<<16u)|(bits>>16u);
    bits=((bits&0x55555555u)<<1u)|((bits&0xAAAAAAAAu)>>1u);
    bits=((bits&0x33333333u)<<2u)|((bits&0xCCCCCCCCu)>>2u);
    bits=((bits&0x0F0F0F0Fu)<<4u)|((bits&0xF0F0F0F0u)>>4u);
    bits=((bits&0x00FF00FFu)<<8u)|((bits&0xFF00FF00u)>>8u);
    return float(bits)*2.3283064365386963e-10;
}
vec2 Hammersley(uint i, uint N){ return vec2(float(i)/float(N), RadicalInverse_VdC(i)); }
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float r){
    float a=r*r; float phi=2.0*PI*Xi.x;
    float cosTheta=sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
    float sinTheta=sqrt(1.0-cosTheta*cosTheta);
    vec3 H=vec3(cos(phi)*sinTheta,sin(phi)*sinTheta,cosTheta);
    vec3 up=abs(N.z)<0.999?vec3(0,0,1):vec3(1,0,0);
    vec3 tangent=normalize(cross(up,N));
    vec3 bitangent=cross(N,tangent);
    return normalize(tangent*H.x+bitangent*H.y+N*H.z);
}
float GeometrySchlickGGX(float NdotV, float r){ float a=r; float k=a*a/2.0; return NdotV/(NdotV*(1.0-k)+k); }
float GeometrySmith(vec3 N, vec3 V, vec3 L, float r){
    return GeometrySchlickGGX(max(dot(N,V),0.0),r)*GeometrySchlickGGX(max(dot(N,L),0.0),r);
}
vec2 IntegrateBRDF(float NdotV, float r){
    vec3 V=vec3(sqrt(1.0-NdotV*NdotV),0.0,NdotV);
    float A=0.0,B=0.0; vec3 N=vec3(0,0,1);
    const uint SAMPLE_COUNT=1024u;
    for(uint i=0u;i<SAMPLE_COUNT;i++){
        vec2 Xi=Hammersley(i,SAMPLE_COUNT);
        vec3 H=ImportanceSampleGGX(Xi,N,r);
        vec3 L=normalize(2.0*dot(V,H)*H-V);
        float NdotL=max(L.z,0.0); float NdotH=max(H.z,0.0); float VdotH=max(dot(V,H),0.0);
        if(NdotL>0.0){
            float G=GeometrySmith(N,V,L,r);
            float G_Vis=G*VdotH/(NdotH*NdotV);
            float Fc=pow(1.0-VdotH,5.0);
            A+=(1.0-Fc)*G_Vis; B+=Fc*G_Vis;
        }
    }
    return vec2(A,B)/float(SAMPLE_COUNT);
}
void main(){ FragColor=IntegrateBRDF(TexCoords.x, TexCoords.y); }
"""

PBR_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
out vec3 FragPos;
out vec3 Normal;
out vec2 UV;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;
void main(){
    FragPos = vec3(model * vec4(aPos,1.0));
    Normal = normalize(normalMatrix * aNormal);
    UV = aUV;
    gl_Position = projection * view * vec4(FragPos,1.0);
}
"""

PBR_FS = """
#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 UV;

uniform sampler2D albedoMap;
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;
uniform vec3 camPos;
uniform float metallic;
uniform float roughness;

const float PI = 3.14159265359;

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float r){
    return F0 + (max(vec3(1.0-r), F0) - F0) * pow(clamp(1.0-cosTheta, 0.0, 1.0), 5.0);
}

void main(){
    vec3 albedo = pow(texture(albedoMap, UV).rgb, vec3(2.2));
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - FragPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = kD * diffuse + specular;
    vec3 color = ambient;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}
"""

SKYBOX_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec3 localPos;
uniform mat4 projection;
uniform mat4 view;
void main(){
    localPos = aPos;
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
"""

SKYBOX_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube envMap;
void main(){
    vec3 color = texture(envMap, localPos).rgb;
    FragColor = vec4(color, 1.0);
}
"""

# ============ 交互 ============

rotation_x = 0.0
rotation_y = 0.0
zoom_offset = 0.0
last_mouse = [0.0, 0.0]
mouse_left = False
mouse_right = False

def mouse_button_callback(window, button, action, mods):
    global mouse_left, mouse_right, last_mouse
    x, y = glfw.get_cursor_pos(window)
    last_mouse = [x, y]
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_left = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        mouse_right = (action == glfw.PRESS)

def cursor_pos_callback(window, x, y):
    global rotation_x, rotation_y, zoom_offset, last_mouse
    dx = x - last_mouse[0]
    dy = y - last_mouse[1]
    if mouse_left:
        rotation_y += dx * 0.3
        rotation_x += dy * 0.3
    if mouse_right:
        zoom_offset -= dy * 0.005
        zoom_offset = max(-1.0, min(3.0, zoom_offset))
    last_mouse = [x, y]

def scroll_callback(window, xoff, yoff):
    global zoom_offset
    zoom_offset += yoff * 0.1
    zoom_offset = max(-1.0, min(3.0, zoom_offset))

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

# ============ OpenGL 工具函数 ============

def compile_shader(vs_src, fs_src):
    vs = shaders.compileShader(vs_src, GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)

def upload_texture_2d(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    data = img.tobytes()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    return tex

def upload_cubemap(faces):
    order = ["px", "nx", "py", "ny", "pz", "nz"]
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
    for i, name in enumerate(order):
        face = faces[name]
        data = face.tobytes()
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_SRGB,
                     face.width, face.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
    return tex

def create_mesh_vao(verts, normals, uvs, faces):
    interleaved = np.hstack([verts, normals, uvs]).astype(np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    stride = 8 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)
    glBindVertexArray(0)
    return vao, len(faces) * 3

def create_cube_vao():
    verts = np.array([
        -1, 1,-1, -1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1, -1, 1,-1,
        -1,-1, 1, -1,-1,-1, -1, 1,-1, -1, 1,-1, -1, 1, 1, -1,-1, 1,
         1,-1,-1,  1,-1, 1,  1, 1, 1,  1, 1, 1,  1, 1,-1,  1,-1,-1,
        -1,-1, 1, -1, 1, 1,  1, 1, 1,  1, 1, 1,  1,-1, 1, -1,-1, 1,
        -1, 1,-1,  1, 1,-1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1, 1,-1,
        -1,-1,-1, -1,-1, 1,  1,-1,-1,  1,-1,-1, -1,-1, 1,  1,-1, 1,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao

def create_quad_vao():
    data = np.array([
        -1,-1,0, 0,0,  1,-1,0, 1,0,  1,1,0, 1,1,
        -1,-1,0, 0,0,  1,1,0, 1,1,  -1,1,0, 0,1,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao

# ============ IBL 预计算 ============

capture_projection = glm.perspective(glm.radians(90.0), 1.0, 0.1, 10.0)
capture_views = [
    glm.lookAt(glm.vec3(0), glm.vec3( 1, 0, 0), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3(-1, 0, 0), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 1, 0), glm.vec3(0, 0, 1)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0,-1, 0), glm.vec3(0, 0,-1)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 0, 1), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 0,-1), glm.vec3(0,-1, 0)),
]

def generate_ibl(env_cubemap, cube_vao, quad_vao):
    fbo = glGenFramebuffers(1)
    rbo = glGenRenderbuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)

    # 1. 辐照度卷积
    print("  计算辐照度贴图...")
    irr_size = 32
    irr_cubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, irr_cubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     irr_size, irr_size, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, irr_size, irr_size)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    irr_shader = compile_shader(CUBE_VS, IRRADIANCE_FS)
    glUseProgram(irr_shader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
    glUniform1i(glGetUniformLocation(irr_shader, "environmentMap"), 0)
    glUniformMatrix4fv(glGetUniformLocation(irr_shader, "projection"), 1, GL_FALSE,
                       glm.value_ptr(capture_projection))
    glViewport(0, 0, irr_size, irr_size)
    for i in range(6):
        glUniformMatrix4fv(glGetUniformLocation(irr_shader, "view"), 1, GL_FALSE,
                           glm.value_ptr(capture_views[i]))
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irr_cubemap, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindVertexArray(cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)

    # 2. 预过滤环境贴图
    print("  计算预过滤环境贴图...")
    pref_size = 128
    pref_cubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, pref_cubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     pref_size, pref_size, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    pref_shader = compile_shader(CUBE_VS, PREFILTER_FS)
    glUseProgram(pref_shader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
    glUniform1i(glGetUniformLocation(pref_shader, "environmentMap"), 0)
    glUniformMatrix4fv(glGetUniformLocation(pref_shader, "projection"), 1, GL_FALSE,
                       glm.value_ptr(capture_projection))

    max_mip = 5
    for mip in range(max_mip):
        mip_size = int(pref_size * (0.5 ** mip))
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mip_size, mip_size)
        glViewport(0, 0, mip_size, mip_size)
        r = mip / (max_mip - 1)
        glUniform1f(glGetUniformLocation(pref_shader, "roughness"), r)
        for i in range(6):
            glUniformMatrix4fv(glGetUniformLocation(pref_shader, "view"), 1, GL_FALSE,
                               glm.value_ptr(capture_views[i]))
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, pref_cubemap, mip)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindVertexArray(cube_vao)
            glDrawArrays(GL_TRIANGLES, 0, 36)

    # 3. BRDF LUT
    print("  计算 BRDF LUT...")
    brdf_size = 512
    brdf_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, brdf_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, brdf_size, brdf_size, 0, GL_RG, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    brdf_shader = compile_shader(BRDF_VS, BRDF_FS)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, brdf_size, brdf_size)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdf_tex, 0)
    glViewport(0, 0, brdf_size, brdf_size)
    glUseProgram(brdf_shader)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(quad_vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [fbo])
    glDeleteRenderbuffers(1, [rbo])

    return irr_cubemap, pref_cubemap, brdf_tex

# ============ 主程序 ============

def main():
    if not glfw.init():
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    w, h = int(cam_res[0]), int(cam_res[1])
    window = glfw.create_window(w, h, "Chair Viewer (PBR + IBL)", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glClearColor(0.15, 0.15, 0.15, 1.0)

    cube_vao = create_cube_vao()
    quad_vao = create_quad_vao()

    print("正在加载环境 cubemap...")
    cube_faces, face_size = load_cubemap_faces("/home/ai/Desktop/Regal_Palace_Ballroom_cubemap")
    env_cubemap = upload_cubemap(cube_faces)
    print(f"环境 cubemap 加载完成 (每面 {face_size}x{face_size})")

    print("正在预计算 IBL...")
    irr_cubemap, pref_cubemap, brdf_tex = generate_ibl(env_cubemap, cube_vao, quad_vao)
    print("IBL 预计算完成！\n")

    pbr_shader = compile_shader(PBR_VS, PBR_FS)
    skybox_shader = compile_shader(SKYBOX_VS, SKYBOX_FS)

    gpu_meshes = []
    for m in mesh_data:
        vao, count = create_mesh_vao(m["vertices"], m["normals"], m["uvs"], m["faces"])
        tex = upload_texture_2d(m["tex_img"])
        gpu_meshes.append({
            "vao": vao, "count": count, "tex": tex,
            "metallic": m["metallic"], "roughness": m["roughness"],
        })

    print(f"相机位置: {cam_pos}, FOV: {cam_fov}")
    print("操作: 左键旋转, 右键/滚轮缩放, Esc退出")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = fb_w / fb_h if fb_h > 0 else 1.0
        projection = glm.perspective(glm.radians(cam_fov), aspect, 0.01, 100.0)
        eye = glm.vec3(cam_pos[0], cam_pos[1], cam_pos[2] + zoom_offset)
        target = glm.vec3(center[0], center[1], center[2])
        view = glm.lookAt(eye, target, glm.vec3(0, 1, 0))

        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(rotation_x), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(rotation_y), glm.vec3(0, 1, 0))
        normal_mat = glm.mat3(glm.transpose(glm.inverse(model)))

        # 绘制椅子 (PBR + IBL)
        glUseProgram(pbr_shader)
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix3fv(glGetUniformLocation(pbr_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_mat))
        glUniform3f(glGetUniformLocation(pbr_shader, "camPos"), eye.x, eye.y, eye.z)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, irr_cubemap)
        glUniform1i(glGetUniformLocation(pbr_shader, "irradianceMap"), 1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_CUBE_MAP, pref_cubemap)
        glUniform1i(glGetUniformLocation(pbr_shader, "prefilterMap"), 2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, brdf_tex)
        glUniform1i(glGetUniformLocation(pbr_shader, "brdfLUT"), 3)

        for gm in gpu_meshes:
            glUniform1f(glGetUniformLocation(pbr_shader, "metallic"), gm["metallic"])
            glUniform1f(glGetUniformLocation(pbr_shader, "roughness"), gm["roughness"])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, gm["tex"])
            glUniform1i(glGetUniformLocation(pbr_shader, "albedoMap"), 0)
            glBindVertexArray(gm["vao"])
            glDrawElements(GL_TRIANGLES, gm["count"], GL_UNSIGNED_INT, None)

        # 绘制天空盒
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_FALSE)
        glUseProgram(skybox_shader)
        skybox_view = view * glm.mat4(model)
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "view"), 1, GL_FALSE, glm.value_ptr(skybox_view))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
        glUniform1i(glGetUniformLocation(skybox_shader, "envMap"), 0)
        glBindVertexArray(cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
