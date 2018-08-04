#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec2[] vsFragTextureUV;
in mat4[] vsModelMatrix;

out vec2 gsFragTextureUV;

// uniforms
uniform mat4 projectionMatrix;
uniform mat4 cameraMatrix;
uniform int applyFoliageAnimation;
uniform int frame;

//
void computeVertexTransform(int vertexIdx, mat4 shaderTransformMatrix) {
	// Emit a primitive only if the sign of the dot product is positive
	mat4 mvMatrix = cameraMatrix * vsModelMatrix[vertexIdx] * shaderTransformMatrix;
	mat4 mvpMatrix = projectionMatrix * cameraMatrix * vsModelMatrix[vertexIdx] * shaderTransformMatrix;
	mat4 normalMatrix = mat4(transpose(inverse(mat3(cameraMatrix * vsModelMatrix[vertexIdx] * shaderTransformMatrix))));
	gsFragTextureUV = vsFragTextureUV[vertexIdx];
	gl_Position = mvpMatrix * gl_in[vertexIdx].gl_Position;
}

void computeVertex(int vertexIdx) {
	// Emit a primitive only if the sign of the dot product is positive
	mat4 mvMatrix = cameraMatrix * vsModelMatrix[vertexIdx];
	mat4 mvpMatrix = projectionMatrix * cameraMatrix * vsModelMatrix[vertexIdx];
	mat4 normalMatrix = mat4(transpose(inverse(mat3(cameraMatrix * vsModelMatrix[vertexIdx]))));
	gsFragTextureUV = vsFragTextureUV[vertexIdx];
	gl_Position = mvpMatrix * gl_in[vertexIdx].gl_Position;
}

{$FUNCTIONS}

void main()
{
	if (applyFoliageAnimation == 1 && length(cameraMatrix * vsModelMatrix[0] * gl_in[0].gl_Position) < 75.0) {
		vec3 verticesMin;
		vec3 verticesMax;
		vec3 verticesCenter;
		for (int i = 0; i < gl_in.length(); i++) {
			vec3 vertex = gl_in[i].gl_Position.xyz;
			if (i == 0) {
				verticesMin = vertex;
				verticesMax = vertex;
			} else {
				for (int j = 0; j < 3; j++) {
					if (vertex[j] < verticesMin[j]) verticesMin[j] = vertex[j];
					if (vertex[j] > verticesMax[j]) verticesMax[j] = vertex[j];
				}
			}
		}
		verticesCenter = verticesMin + ((verticesMax - verticesMin) / 2.0);

		float zAxisRotation = (frame + int(verticesCenter[0] * 8.0 * (verticesCenter[0] < 0.0?-1.0:1.0))) % 160;
		zAxisRotation = (((zAxisRotation < 160 / 2?zAxisRotation:160 - zAxisRotation) - (160 / 4)) * 1.0f / 8.0f);
		mat4 shaderTransformMatrix =
			createTranslationMatrix(vec3(verticesCenter[0], verticesMin[1], verticesCenter[2])) *
			createRotationMatrix(vec3(0.0, 0.0, 1.0), zAxisRotation) *
			createTranslationMatrix(vec3(verticesCenter[0], verticesMin[1], verticesCenter[2]) * -1.0);

		for (int i = 0; i < gl_in.length() / 3; i++) {
			computeVertexTransform(i * 3 + 0, shaderTransformMatrix);
			EmitVertex();
			computeVertexTransform(i * 3 + 1, shaderTransformMatrix);
			EmitVertex();
			computeVertexTransform(i * 3 + 2, shaderTransformMatrix);
			EmitVertex();
			EndPrimitive();
		}
	} else {
		for (int i = 0; i < gl_in.length() / 3; i++) {
			computeVertex(i * 3 + 0);
			EmitVertex();
			computeVertex(i * 3 + 1);
			EmitVertex();
			computeVertex(i * 3 + 2);
			EmitVertex();
			EndPrimitive();
		}
	}
}
