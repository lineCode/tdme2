#include <tdme/engine/subsystems/shadowmapping/ShadowMappingShaderPreBaseImplementation.h>

#include <tdme/engine/subsystems/lighting/LightingShader.h>
#include <tdme/engine/subsystems/lighting/LightingShaderConstants.h>
#include <tdme/engine/subsystems/renderer/GLRenderer.h>
#include <tdme/math/Matrix4x4.h>
#include <tdme/os/filesystem/FileSystem.h>
#include <tdme/os/filesystem/FileSystemInterface.h>
#include <tdme/utils/Console.h>

using tdme::engine::subsystems::shadowmapping::ShadowMappingShaderPreBaseImplementation;
using tdme::engine::subsystems::lighting::LightingShader;
using tdme::engine::subsystems::lighting::LightingShaderConstants;
using tdme::engine::subsystems::renderer::GLRenderer;
using tdme::math::Matrix4x4;
using tdme::os::filesystem::FileSystem;
using tdme::os::filesystem::FileSystemInterface;
using tdme::utils::Console;

ShadowMappingShaderPreBaseImplementation::ShadowMappingShaderPreBaseImplementation(GLRenderer* renderer)
{
	this->renderer = renderer;
	initialized = false;
}

ShadowMappingShaderPreBaseImplementation::~ShadowMappingShaderPreBaseImplementation() {
}

bool ShadowMappingShaderPreBaseImplementation::isInitialized()
{
	return initialized;
}

void ShadowMappingShaderPreBaseImplementation::initialize()
{
	auto rendererVersion = renderer->getGLVersion();

	// load shadow mapping shaders
	//	pre render
	vertexShaderGlId = renderer->loadShader(
		renderer->SHADER_VERTEX_SHADER,
		"shader/" + rendererVersion + "/shadowmapping",
		"pre_vertexshader.c"
	);
	if (vertexShaderGlId == 0) return;
	if (renderer->isGeometryShaderAvailable() == true) {
		geometryShaderGlId = renderer->loadShader(
			renderer->SHADER_GEOMETRY_SHADER,
			"shader/" + rendererVersion + "/shadowmapping",
			"pre_geometryshader.c",
			"",
			/*
			FileSystem::getInstance()->getContentAsString(
				"shader/" + rendererVersion + "/lighting",
				"render_computevertex.inc.c"
			) +
			"\n\n" +
			*/
			FileSystem::getInstance()->getContentAsString(
				"shader/" + rendererVersion + "/functions",
				"create_rotation_matrix.inc.c"
			) +
			"\n\n" +
			FileSystem::getInstance()->getContentAsString(
				"shader/" + rendererVersion + "/functions",
				"create_translation_matrix.inc.c"
			)
		);
		if (geometryShaderGlId == 0) return;
	}
	fragmentShaderGlId = renderer->loadShader(
		renderer->SHADER_FRAGMENT_SHADER,
		"shader/" + rendererVersion + "/shadowmapping",
		"pre_fragmentshader.c"
	);
	if (fragmentShaderGlId == 0) return;

	// create shadow mapping render program
	//	pre
	programGlId = renderer->createProgram();
	renderer->attachShaderToProgram(programGlId, vertexShaderGlId);
	if (renderer->isGeometryShaderAvailable() == true) {
		renderer->attachShaderToProgram(programGlId, geometryShaderGlId);
	}
	renderer->attachShaderToProgram(programGlId, fragmentShaderGlId);
	// map inputs to attributes
	if (renderer->isUsingProgramAttributeLocation() == true) {
		renderer->setProgramAttributeLocation(programGlId, 0, "inVertex");
		renderer->setProgramAttributeLocation(programGlId, 2, "inTextureUV");
	}
	// link
	if (renderer->linkProgram(programGlId) == false) return;

	// uniforms
	if (renderer->isInstancedRenderingAvailable() == true) {
		//	uniforms
		uniformProjectionMatrix = renderer->getProgramUniformLocation(programGlId, "projectionMatrix");
		if (uniformProjectionMatrix == -1) return;
		uniformCameraMatrix = renderer->getProgramUniformLocation(programGlId, "cameraMatrix");
		if (uniformCameraMatrix == -1) return;
	} else {
		//	uniforms
		uniformMVPMatrix = renderer->getProgramUniformLocation(programGlId, "mvpMatrix");
		if (uniformMVPMatrix == -1) return;
	}
	uniformTextureMatrix = renderer->getProgramUniformLocation(programGlId, "textureMatrix");
	if (uniformTextureMatrix == -1) return;
	uniformDiffuseTextureUnit = renderer->getProgramUniformLocation(programGlId, "diffuseTextureUnit");
	if (uniformDiffuseTextureUnit == -1) return;
	uniformDiffuseTextureAvailable = renderer->getProgramUniformLocation(programGlId, "diffuseTextureAvailable");
	if (uniformDiffuseTextureAvailable == -1) return;
	uniformDiffuseTextureMaskedTransparency = renderer->getProgramUniformLocation(programGlId, "diffuseTextureMaskedTransparency");
	if (uniformDiffuseTextureMaskedTransparency == -1) return;
	uniformDiffuseTextureMaskedTransparencyThreshold = renderer->getProgramUniformLocation(programGlId, "diffuseTextureMaskedTransparencyThreshold");
	if (uniformDiffuseTextureMaskedTransparencyThreshold == -1) return;

	//
	if (renderer->isGeometryShaderAvailable() == true) {
		uniformFrame = renderer->getProgramUniformLocation(programGlId, "frame");
		if (uniformFrame == -1) return;
	}

	//
	initialized = true;
}

void ShadowMappingShaderPreBaseImplementation::useProgram()
{
	renderer->useProgram(programGlId);
	if (renderer->isGeometryShaderAvailable() == true) {
		renderer->setProgramUniformInteger(uniformFrame, renderer->frame);
	}
}

void ShadowMappingShaderPreBaseImplementation::unUseProgram()
{
}

void ShadowMappingShaderPreBaseImplementation::updateMatrices(const Matrix4x4& mvpMatrix)
{
	if (renderer->isInstancedRenderingAvailable() == true) {
		renderer->setProgramUniformFloatMatrix4x4(uniformProjectionMatrix, renderer->getProjectionMatrix().getArray());
		renderer->setProgramUniformFloatMatrix4x4(uniformCameraMatrix, renderer->getCameraMatrix().getArray());
	} else {
		renderer->setProgramUniformFloatMatrix4x4(uniformMVPMatrix, mvpMatrix.getArray());
	}
}

void ShadowMappingShaderPreBaseImplementation::updateTextureMatrix(GLRenderer* renderer) {
	renderer->setProgramUniformFloatMatrix3x3(uniformTextureMatrix, renderer->getTextureMatrix().getArray());
}

void ShadowMappingShaderPreBaseImplementation::updateMaterial(GLRenderer* renderer)
{
	renderer->setProgramUniformInteger(uniformDiffuseTextureMaskedTransparency, renderer->material.diffuseTextureMaskedTransparency);
	renderer->setProgramUniformFloat(uniformDiffuseTextureMaskedTransparencyThreshold, renderer->material.diffuseTextureMaskedTransparencyThreshold);
}

void ShadowMappingShaderPreBaseImplementation::bindTexture(GLRenderer* renderer, int32_t textureId)
{
	switch (renderer->getTextureUnit()) {
		case LightingShaderConstants::TEXTUREUNIT_DIFFUSE:
			renderer->setProgramUniformInteger(uniformDiffuseTextureAvailable, textureId == 0 ? 0 : 1);
			break;
	}
}
