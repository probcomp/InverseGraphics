import Revise
using ModernGL
import GLFW
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN

# +
################
# shader utils #
################

# from https://github.com/JuliaGL/ModernGL.jl/blob/d56e4ad51f4459c97deeea7666361600a1e6065e/test/util.jl

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

##################
# OpenGL shaders #
##################

# vertex shader for computing depth image
const vertex_source = """
#version 330 core
uniform mat4 mvp;
in vec3 position;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
}
"""

# fragment shader for sillhouette
const fragment_source = """
# version 330 core
out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""


function make_compute_depth_shader()
    vertex_shader = createShader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = createShader(fragment_source, GL_FRAGMENT_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glBindFragDataLocation(shader_program, 0, "outColor")
    glLinkProgram(shader_program)
    pos_attr = glGetAttribLocation(shader_program, "position")
    (shader_program, pos_attr)
end

# +
function pose_to_matrix(pose::Pose)::Matrix{Float32}
    R = Matrix{Float32}(pose.orientation)
    mat = zeros(Float32,4,4)
    mat[end,end] = 1
    mat[1:3,1:3] = R
    mat[1:3,4] = pose.pos
    return mat
end

function I4(t)
    x = zeros(t, 4,4)
    x[1,1] = 1.0
    x[2,2] = 1.0
    x[3,3] = 1.0
    x[4,4] = 1.0
    x
end

function compute_projection_matrix(fx, fy, cx, cy, near, far, skew=0f0)
    proj = I4(Float32)
    proj[1, 1] = fx
    proj[2, 2] = fy
    proj[1, 2] = skew
    proj[1, 3] = -cx
    proj[2, 3] = -cy
    proj[3, 3] = near + far
    proj[3, 4] = near * far
    proj[4, 4] = 0.0f0
    proj[4, 3] = -1f0
    return proj
end



function compute_ortho_matrix(left, right, bottom, top, near, far)
    ortho = I4(Float32)
    ortho[1, 1] = 2f0 / (right-left)
    ortho[2, 2] = 2f0 / (top-bottom)
    ortho[3, 3] = - 2f0 / (far - near)
    ortho[1, 4] = - (right + left) / (right - left)
    ortho[2, 4] = - (top + bottom) / (top - bottom)
    ortho[3, 4] = - (far + near) / (far - near)
    return ortho
end

function perspective_matrix(width, height, fx, fy, cx, cy, near, far)
    # (height-cy) is used instead of cy because of the difference between
    # image coordinate systems between OpenCV and OpenGL. In the former,
    # the origin is at the top-left of the image while in the latter the
    # origin is at the bottom-left.
    proj_matrix = compute_projection_matrix(
            fx, fy, cx, (height-cy),
            near, far, 0.f0)
    ndc_matrix = compute_ortho_matrix(0, width, 0, height, near, far)
    ndc_matrix * proj_matrix
end

cam = (width=20,height=20, near=0.001, far=5.0, fx=20.0, fy=20.0, cx=10.0, cy= 10.0)

# -

p = perspective_matrix(cam.width, cam.height, cam.fx, cam.fy,
    cam.cx, cam.cy,
    cam.near, cam.far
)

window_hint = [
    (GLFW.SAMPLES,      0),
    (GLFW.DEPTH_BITS,   24),
    (GLFW.ALPHA_BITS,   8),
    (GLFW.RED_BITS,     8),
    (GLFW.GREEN_BITS,   8),
    (GLFW.BLUE_BITS,    8),
    (GLFW.STENCIL_BITS, 0),
    (GLFW.AUX_BUFFERS,  0),
    (GLFW.CONTEXT_VERSION_MAJOR, 3),
    (GLFW.CONTEXT_VERSION_MINOR, 3),
    (GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE),
    (GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE),
]
for (key, value) in window_hint
    GLFW.WindowHint(key, value)
end
window = GLFW.CreateWindow(cam.width, cam.height, "DepthRenderer")
GLFW.MakeContextCurrent(window)
compute_depth_shader, pos_attr = make_compute_depth_shader()
glEnable(GL_DEPTH_TEST)
glViewport(0, 0, cam.width, cam.height)
glClear(GL_DEPTH_BUFFER_BIT)

all_vaos = []

# +
a = Float32[-1, -1, -2]
b = Float32[1, -1, -2]
c = Float32[0, 1, -2]
vertices = hcat(a, b, c)
indices = hcat(UInt32[0, 1, 2])
mesh = (vertices=vertices, indices=indices)

vao = Ref(GLuint(0))
glGenVertexArrays(1, vao)
glBindVertexArray(vao[])

# copy vertex data into an OpenGL buffer
vbo = Ref(GLuint(0))
glGenBuffers(1, vbo)
glBindBuffer(GL_ARRAY_BUFFER, vbo[])
glBufferData(GL_ARRAY_BUFFER, sizeof(mesh.vertices), Ref(mesh.vertices, 1), GL_STATIC_DRAW)

# element buffer object for indices
ebo = Ref(GLuint(0))
glGenBuffers(1, ebo)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(mesh.indices), Ref(mesh.indices, 1), GL_STATIC_DRAW)

# set vertex attribute pointers
glVertexAttribPointer(pos_attr, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), C_NULL)
glEnableVertexAttribArray(pos_attr)

# unbind it
glBindVertexArray(0)

push!(all_vaos,vao[])
# -

view = pose_to_matrix(IDENTITY_POSE)

model  = pose_to_matrix(IDENTITY_POSE)

# +

function scale_depth(x, near, far)
    far .* near ./ (far .- (far .- near) .* x)
end


# +
vao = all_vaos[1]
mvp = p * view * model
glUseProgram(compute_depth_shader)
glUniformMatrix4fv(0, 1, GL_FALSE, Ref(mvp, 1))
glBindVertexArray(vao)
glDrawElements(GL_TRIANGLES, size(mesh.indices)[2] * 3, GL_UNSIGNED_INT, C_NULL)
glBindVertexArray(0)

data = Matrix{Float32}(undef, cam.width, cam.height)
glReadPixels(0, 0, cam.width, cam.height, GL_DEPTH_COMPONENT, GL_FLOAT, Ref(data, 1))

depth_image = scale_depth(data, cam.near, cam.far)
# -

import Images as I
# Viewing images
function view_depth_image(depth_image)
    img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
    I.convert.(I.RGBA, img)
end


view_depth_image(depth_image)




