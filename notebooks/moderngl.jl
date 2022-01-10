using ModernGL
import GLFW
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN

original_camera = (width=640,height=480)

window = GLFW.CreateWindow(original_camera.width, original_camera.height, "OpenGL Example")
GLFW.MakeContextCurrent(window)
GLFW.ShowWindow(window)
GLFW.SetWindowSize(window, original_camera.width, original_camera.height) # Seems to be necessary to guarantee that window > 0

glEnable(GL_DEPTH_TEST)
glViewport(0, 0, original_camera.width, original_camera.height)
glClear(GL_DEPTH_BUFFER_BIT)

# +
const vertexShader_depth = """
#version 460
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
layout (location=0) in vec3 position;
void main() {
    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
}
"""
const fragmentShader_depth = """
#version 460
out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

function validateShader(shader)
	success = GLint[0]
	glGetShaderiv(shader, GL_COMPILE_STATUS, success)
	success[] == GL_TRUE
end

function glErrorMessage()
# Return a string representing the current OpenGL error flag, or the empty string if there's no error.
	err = glGetError()
	err == GL_NO_ERROR ? "" :
	err == GL_INVALID_ENUM ? "GL_INVALID_ENUM: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_VALUE ? "GL_INVALID_VALUE: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_FRAMEBUFFER_OPERATION ? "GL_INVALID_FRAMEBUFFER_OPERATION: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_OUT_OF_MEMORY ? "GL_OUT_OF_MEMORY: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded." : "Unknown OpenGL error with error code $err."
end

function getInfoLog(obj::GLuint)
	# Return the info log for obj, whether it be a shader or a program.
	isShader = glIsShader(obj)
	getiv = isShader == GL_TRUE ? glGetShaderiv : glGetProgramiv
	getInfo = isShader == GL_TRUE ? glGetShaderInfoLog : glGetProgramInfoLog
	# Get the maximum possible length for the descriptive error message
	len = GLint[0]
	getiv(obj, GL_INFO_LOG_LENGTH, len)
	maxlength = len[]
	# TODO: Create a macro that turns the following into the above:
	# maxlength = @glPointer getiv(obj, GL_INFO_LOG_LENGTH, GLint)
	# Return the text of the message if there is any
	if maxlength > 0
		buffer = zeros(GLchar, maxlength)
		sizei = GLsizei[0]
		getInfo(obj, maxlength, sizei, buffer)
		len = sizei[]
		unsafe_string(pointer(buffer), len)
	else
		""
	end
end

function createShader(source, typ)

    # Create the shader
	shader = glCreateShader(typ)::GLuint
	if shader == 0
		error("Error creating shader: ", glErrorMessage())
	end

	# Compile the shader
	glShaderSource(
        shader, 1, convert(Ptr{UInt8},
        pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
	glCompileShader(shader)

	# Check for errors
	!validateShader(shader) && error("Shader creation error: ", getInfoLog(shader))
	shader
end

vertex_shader = createShader(vertexShader_depth, GL_VERTEX_SHADER)
fragment_shader = createShader(fragmentShader_depth, GL_FRAGMENT_SHADER)
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glBindFragDataLocation(shader_program, 0, "outColor")
glLinkProgram(shader_program)

# -

a = Float32[-0.25, -0.25, -1]
b = Float32[0.25, -0.25, -1]
c = Float32[0.0, 0.25, -1]
vertices = hcat(a, b, c)
indices = hcat(UInt32[0, 1, 2])

all_vaos = []

# +


vao = Ref(GLuint(0))
glGenVertexArrays(1, vao)
glBindVertexArray(vao[])

# copy vertex data into an OpenGL buffer
vbo = Ref(GLuint(0))
glGenBuffers(1, vbo)
glBindBuffer(GL_ARRAY_BUFFER, vbo[])
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), Ref(vertices, 1), GL_STATIC_DRAW)

# element buffer object for indices
ebo = Ref(GLuint(0))
glGenBuffers(1, ebo)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), Ref(indices, 1), GL_STATIC_DRAW)

pos_attr = glGetAttribLocation(shader_program, "position")

# set vertex attribute pointers
glVertexAttribPointer(pos_attr, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), C_NULL)
glEnableVertexAttribArray(pos_attr)

# unbind it
glBindVertexArray(0)

push!(all_vaos, vao[])
# -

glClearColor(1.0, 1.0, 1.0, 1)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
glEnable(GL_DEPTH_TEST)


glUseProgram(shader_program)

V = pose

# +


GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'V'), 1, GL.GL_TRUE,
                      self.V)
GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'P'), 1, GL.GL_FALSE,
                      self.P)
GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'pose_trans'), 1,
                      GL.GL_FALSE, trans)
GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'pose_rot'), 1,
                      GL.GL_TRUE, rot)
# -

MGL.shaders

depth_tex = Ref(GL.GLuint(0))
GL.glGenTextures(1, depth_tex)

GL.glBindTexture(GL.GL_TEXTURE_2D, depth_tex[])

a = GL.glTexImage2D

show_depth_vao = Ref(GL.GLuint(0))
GL.glGenVertexArrays(1, show_depth_vao)
GL.glBindVertexArray(show_depth_vao[])

const vertexShader_depth = """
#version 460
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
layout (location=0) in vec3 position;
void main() {
    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
}
"""
const fragmentShader_depth = """
#version 460
out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)



?GL.glGenFramebuffers

import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import OpenCV as CV
try
    import MeshCatViz as V
catch
    import MeshCatViz as V
end

using ModernGL
import GLFW

V.setup_visualizer()







# +

renderer = GL.setup_renderer(original_camera, GL.SegmentationMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)

for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f
    )
end

# # +
idx = 4
# id = ids[1]
p = gt_poses[1]
colors = map(I.RGBA,I.distinguishable_colors(length(ids)))

# +
window = GLFW.CreateWindow(camera.width, camera.height, "OpenGL Example")
GLFW.MakeContextCurrent(window)
GLFW.ShowWindow(window)
GLFW.SetWindowSize(window,camera.width, camera.height) # Seems to be necessary to guarantee that window > 0



# -







const vertexShader_depth = """
#version 460
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
layout (location=0) in vec3 position;
void main() {
    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
}
"""
const fragmentShader_depth = """
#version 460
out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""



    glViewport(0, 0, camera.width, camera.height)




rgb_image_p = permutedims(rgb_image,(3,1,2));
rgb_edges = CV.Canny(rgb_image_p, 50.0,600.0)
GL.view_depth_image(rgb_edges[1,:,:])

d = reshape(gt_depth_image, (1,size(gt_depth_image)...))
m = 200.0
d = round.(UInt8,clamp.(d,0.0, m) ./ m .* 255.0)
GL.view_depth_image(d[1,:,:])

d_edges = CV.Canny(reshape(d, (1,size(gt_depth_image)...)), 1.0,500.0)
GL.view_depth_image(d_edges[1,:,:])

GL.view_depth_image(diff(d;dims=2)[1,:,:])


GL.view_depth_image(diff(diff(d;dims=3);dims=3)[1,:,:])



# +
normalize(v) = (v./sqrt(sum(v.^2)))

c = GL.depth_image_to_point_cloud(gt_depth_image, camera; flatten=false)
h,w = size(c)[1:2]
m = 5
dot_prod = zeros(h,w)
for i in (1+m):(h-m)
    for j in (1+m):(w-m)
        v1 = normalize(c[i-m,j,:] .- c[i,j,:])
        v2 = normalize(c[i,j,:] .- c[i+m,j,:])
        dot_prod[i,j] = sum(v1 .* v2)
    end
end
dot_prod
# -

GL.view_depth_image(dot_prod .< 0.01)

c = GL.depth_image_to_point_cloud(gt_depth_image, camera; flatten=false)
h,w = size(c)[1:2]
m = 5
dot_prod = zeros(h,w)
for i in (1+m):(h-m)
    for j in (1+m):(w-m)
        v1 = normalize(c[i,j-m,:] .- c[i,j,:])
        v2 = normalize(c[i,j,:] .- c[i,j+m,:])
        dot_prod[i,j] = sum(v1 .* v2)
    end
end
dot_prod

GL.view_depth_image(dot_prod .> 0.6)

gt_cloud = T.move_points_to_frame_b(GL.depth_image_to_point_cloud(gt_depth_image, camera),cam_pose)
V.viz(T.voxelize(gt_cloud,0.2))


