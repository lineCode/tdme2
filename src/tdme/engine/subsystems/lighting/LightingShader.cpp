#include <tdme/engine/subsystems/lighting/LightingShader.h>

#include <algorithm>
#include <string>

#include <tdme/engine/subsystems/renderer/GLRenderer_Light.h>
#include <tdme/engine/subsystems/renderer/GLRenderer_Material.h>
#include <tdme/engine/subsystems/renderer/GLRenderer.h>
#include <tdme/math/Matrix4x4.h>

using std::copy;
using std::begin;
using std::end;
using std::to_string;
using std::string;

using tdme::engine::subsystems::lighting::LightingShader;
using tdme::engine::subsystems::renderer::GLRenderer_Light;
using tdme::engine::subsystems::renderer::GLRenderer_Material;
using tdme::engine::subsystems::renderer::GLRenderer;
using tdme::math::Matrix4x4;

LightingShader::LightingShader(GLRenderer* renderer) 
{
	this->renderer = renderer;
	isRunning = false;
	initialized = false;
}

constexpr int32_t LightingShader::MAX_LIGHTS;
constexpr int32_t LightingShader::TEXTUREUNIT_DIFFUSE;
constexpr int32_t LightingShader::TEXTUREUNIT_SPECULAR;
constexpr int32_t LightingShader::TEXTUREUNIT_DISPLACEMENT;
constexpr int32_t LightingShader::TEXTUREUNIT_NORMAL;

bool LightingShader::isInitialized()
{
	return initialized;
}

void LightingShader::initialize()
{
	auto rendererVersion = renderer->getGLVersion();

	// lighting
	//	fragment shader
	renderLightingFragmentShaderId = renderer->loadShader(
		renderer->SHADER_FRAGMENT_SHADER,
		"shader/" + rendererVersion + "/lighting",
		"render_fragmentshader.c"
	);
	if (renderLightingFragmentShaderId == 0)
		return;

	//	vertex shader
	renderLightingVertexShaderId = renderer->loadShader(
		renderer->SHADER_VERTEX_SHADER,
		"shader/" + rendererVersion + "/lighting",
		"render_vertexshader.c"
	);
	if (renderLightingVertexShaderId == 0)
		return;

	// create, attach and link program
	renderLightingProgramId = renderer->createProgram();
	renderer->attachShaderToProgram(renderLightingProgramId, renderLightingVertexShaderId);
	renderer->attachShaderToProgram(renderLightingProgramId, renderLightingFragmentShaderId);

	// map inputs to attributes
	if (renderer->isUsingProgramAttributeLocation() == true) {
		renderer->setProgramAttributeLocation(renderLightingProgramId, 0, "inVertex");
		renderer->setProgramAttributeLocation(renderLightingProgramId, 1, "inNormal");
		renderer->setProgramAttributeLocation(renderLightingProgramId, 2, "inTextureUV");
	}

	// link program
	if (renderer->linkProgram(renderLightingProgramId) == false)
		return;

	// get uniforms
	//	globals
	uniformDiffuseTextureUnit = renderer->getProgramUniformLocation(renderLightingProgramId, "diffuseTextureUnit");
	if (uniformDiffuseTextureUnit == -1)
		return;

	uniformDiffuseTextureAvailable = renderer->getProgramUniformLocation(renderLightingProgramId, "diffuseTextureAvailable");
	if (uniformDiffuseTextureAvailable == -1)
		return;

	uniformDiffuseTextureMaskedTransparency = renderer->getProgramUniformLocation(renderLightingProgramId, "diffuseTextureMaskedTransparency");
	if (uniformDiffuseTextureMaskedTransparency == -1)
		return;

	if (renderer->isDisplacementMappingAvailable() == true) {
		uniformDisplacementTextureUnit = renderer->getProgramUniformLocation(renderLightingProgramId, "displacementTextureUnit");
		if (uniformDisplacementTextureUnit == -1)
			return;

		uniformDisplacementTextureAvailable = renderer->getProgramUniformLocation(renderLightingProgramId, "displacementTextureAvailable");
		if (uniformDisplacementTextureAvailable == -1)
			return;

	}
	if (renderer->isSpecularMappingAvailable()) {
		uniformSpecularTextureUnit = renderer->getProgramUniformLocation(renderLightingProgramId, "specularTextureUnit");
		if (uniformSpecularTextureUnit == -1)
			return;

		uniformSpecularTextureAvailable = renderer->getProgramUniformLocation(renderLightingProgramId, "specularTextureAvailable");
		if (uniformSpecularTextureAvailable == -1)
			return;

	}
	if (renderer->isNormalMappingAvailable()) {
		uniformNormalTextureUnit = renderer->getProgramUniformLocation(renderLightingProgramId, "normalTextureUnit");
		if (uniformNormalTextureUnit == -1)
			return;

		uniformNormalTextureAvailable = renderer->getProgramUniformLocation(renderLightingProgramId, "normalTextureAvailable");
		if (uniformNormalTextureAvailable == -1)
			return;

	}

	// matrices as uniform only if not using instanced rendering
	if (renderer->isInstancedRenderingAvailable() == false) {
		uniformMVPMatrix = renderer->getProgramUniformLocation(renderLightingProgramId, "mvpMatrix");
		if (uniformMVPMatrix == -1)
			return;

		uniformMVMatrix = renderer->getProgramUniformLocation(renderLightingProgramId, "mvMatrix");
		if (uniformMVMatrix == -1)
			return;

		uniformNormalMatrix = renderer->getProgramUniformLocation(renderLightingProgramId, "normalMatrix");
		if (uniformNormalMatrix == -1)
			return;
	} else {
		uniformProjectionMatrix = renderer->getProgramUniformLocation(renderLightingProgramId, "projectionMatrix");
		if (uniformProjectionMatrix == -1)
			return;
	}

	uniformSceneColor = renderer->getProgramUniformLocation(renderLightingProgramId, "sceneColor");
	if (uniformSceneColor == -1)
		return;

	// color effects as uniforms only if not using instanced rendering
	if (renderer->isInstancedRenderingAvailable() == false) {
		uniformEffectColorMul = renderer->getProgramUniformLocation(renderLightingProgramId, "effectColorMul");
		if (uniformEffectColorMul == -1)
			return;

		uniformEffectColorAdd = renderer->getProgramUniformLocation(renderLightingProgramId, "effectColorAdd");
		if (uniformEffectColorAdd == -1)
			return;
	}

	//	material
	uniformMaterialAmbient = renderer->getProgramUniformLocation(renderLightingProgramId, "material.ambient");
	if (uniformMaterialAmbient == -1)
		return;

	uniformMaterialDiffuse = renderer->getProgramUniformLocation(renderLightingProgramId, "material.diffuse");
	if (uniformMaterialDiffuse == -1)
		return;

	uniformMaterialSpecular = renderer->getProgramUniformLocation(renderLightingProgramId, "material.specular");
	if (uniformMaterialSpecular == -1)
		return;

	uniformMaterialEmission = renderer->getProgramUniformLocation(renderLightingProgramId, "material.emission");
	if (uniformMaterialEmission == -1)
		return;

	uniformMaterialShininess = renderer->getProgramUniformLocation(renderLightingProgramId, "material.shininess");
	if (uniformMaterialShininess == -1)
		return;

	//	lights
	for (auto i = 0; i < MAX_LIGHTS; i++) {
		uniformLightEnabled[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) +"].enabled");
		if (uniformLightEnabled[i] == -1)
			return;

		uniformLightAmbient[i] = renderer->getProgramUniformLocation(renderLightingProgramId,"lights[" + to_string(i) + "].ambient");
		if (uniformLightAmbient[i] == -1)
			return;

		uniformLightDiffuse[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].diffuse");
		if (uniformLightDiffuse[i] == -1)
			return;

		uniformLightSpecular[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].specular");
		if (uniformLightSpecular[i] == -1)
			return;

		uniformLightPosition[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].position");
		if (uniformLightPosition[i] == -1)
			return;

		uniformLightSpotDirection[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].spotDirection");
		if (uniformLightSpotDirection[i] == -1)
			return;

		uniformLightSpotExponent[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].spotExponent");
		if (uniformLightSpotExponent[i] == -1)
			return;

		uniformLightSpotCosCutoff[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].spotCosCutoff");
		if (uniformLightSpotCosCutoff[i] == -1)
			return;

		uniformLightConstantAttenuation[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].constantAttenuation");
		if (uniformLightConstantAttenuation[i] == -1)
			return;

		uniformLightLinearAttenuation[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].linearAttenuation");
		if (uniformLightLinearAttenuation[i] == -1)
			return;

		uniformLightQuadraticAttenuation[i] = renderer->getProgramUniformLocation(renderLightingProgramId, "lights[" + to_string(i) + "].quadraticAttenuation");
		if (uniformLightQuadraticAttenuation[i] == -1)
			return;

	}

	//
	initialized = true;
}

void LightingShader::useProgram()
{
	isRunning = true;
	renderer->useProgram(renderLightingProgramId);
	// initialize static uniforms
	if (renderer->isInstancedRenderingAvailable() == true) {
		renderer->setProgramUniformFloatMatrix4x4(uniformProjectionMatrix, renderer->getProjectionMatrix().getArray());
	}
	renderer->setProgramUniformInteger(uniformDiffuseTextureUnit, TEXTUREUNIT_DIFFUSE);
	if (renderer->isSpecularMappingAvailable() == true) {
		renderer->setProgramUniformInteger(uniformSpecularTextureUnit, TEXTUREUNIT_SPECULAR);
	}
	if (renderer->isNormalMappingAvailable() == true) {
		renderer->setProgramUniformInteger(uniformNormalTextureUnit, TEXTUREUNIT_NORMAL);
	}
	if (renderer->isDisplacementMappingAvailable() == true) {
		renderer->setProgramUniformInteger(uniformDisplacementTextureUnit, TEXTUREUNIT_DISPLACEMENT);
	}
	renderer->setProgramUniformFloatVec4(uniformSceneColor, defaultSceneColor);
	// initialize dynamic uniforms
	updateEffect(renderer);
	updateMaterial(renderer);
	for (auto i = 0; i < MAX_LIGHTS; i++) {
		updateLight(renderer, i);
	}
}

void LightingShader::unUseProgram()
{
	isRunning = false;
}

void LightingShader::updateEffect(GLRenderer* renderer)
{
	// skip if not running
	if (isRunning == false) return;

	// skip if using instanced rendering
	if (renderer->isInstancedRenderingAvailable() == true) return;

	//
	renderer->setProgramUniformFloatVec4(uniformEffectColorMul, renderer->effectColorMul);
	renderer->setProgramUniformFloatVec4(uniformEffectColorAdd, renderer->effectColorAdd);
}

void LightingShader::updateMaterial(GLRenderer* renderer)
{
	// skip if not running
	if (isRunning == false)
		return;

	// we dont have alpha on ambient, specular, emission
	tmpColor4[3] = 0.0f;

	// ambient without alpha, as we only use alpha from diffuse color
	copy(begin(renderer->material.ambient), end(renderer->material.ambient), begin(tmpColor4));
	renderer->setProgramUniformFloatVec4(uniformMaterialAmbient, tmpColor4);
	// diffuse
	renderer->setProgramUniformFloatVec4(uniformMaterialDiffuse, renderer->material.diffuse);
	// specular without alpha, as we only use alpha from diffuse color
	copy(begin(renderer->material.specular), end(renderer->material.specular), begin(tmpColor4));
	// emission without alpha, as we only use alpha from diffuse color
	renderer->setProgramUniformFloatVec4(uniformMaterialSpecular, tmpColor4);
	copy(begin(renderer->material.emission), end(renderer->material.emission), begin(tmpColor4));
	renderer->setProgramUniformFloatVec4(uniformMaterialEmission, tmpColor4);
	// shininess
	renderer->setProgramUniformFloat(uniformMaterialShininess, renderer->material.shininess);
	// diffuse texture masked transparency
	renderer->setProgramUniformInteger(uniformDiffuseTextureMaskedTransparency, renderer->material.diffuseTextureMaskedTransparency);
}

void LightingShader::updateLight(GLRenderer* renderer, int32_t lightId)
{
	// skip if not running
	if (isRunning == false)
		return;

	// lighs
	renderer->setProgramUniformInteger(uniformLightEnabled[lightId], renderer->lights[lightId].enabled);
	if (renderer->lights[lightId].enabled == 1) {
		renderer->setProgramUniformFloatVec4(uniformLightAmbient[lightId], renderer->lights[lightId].ambient);
		renderer->setProgramUniformFloatVec4(uniformLightDiffuse[lightId], renderer->lights[lightId].diffuse);
		renderer->setProgramUniformFloatVec4(uniformLightSpecular[lightId], renderer->lights[lightId].specular);
		renderer->setProgramUniformFloatVec4(uniformLightPosition[lightId], renderer->lights[lightId].position);
		renderer->setProgramUniformFloatVec3(uniformLightSpotDirection[lightId], renderer->lights[lightId].spotDirection);
		renderer->setProgramUniformFloat(uniformLightSpotExponent[lightId], renderer->lights[lightId].spotExponent);
		renderer->setProgramUniformFloat(uniformLightSpotCosCutoff[lightId], renderer->lights[lightId].spotCosCutoff);
		renderer->setProgramUniformFloat(uniformLightConstantAttenuation[lightId], renderer->lights[lightId].constantAttenuation);
		renderer->setProgramUniformFloat(uniformLightLinearAttenuation[lightId], renderer->lights[lightId].linearAttenuation);
		renderer->setProgramUniformFloat(uniformLightQuadraticAttenuation[lightId], renderer->lights[lightId].quadraticAttenuation);
	}
}

void LightingShader::updateMatrices(GLRenderer* renderer)
{
	// skip if not running
	if (isRunning == false) return;

	// skip if using instanced rendering
	if (renderer->isInstancedRenderingAvailable() == true) return;

	// model view matrix
	mvMatrix.set(renderer->getModelViewMatrix());
	// object to screen matrix
	mvpMatrix.set(mvMatrix).multiply(renderer->getProjectionMatrix());
	// normal matrix
	normalMatrix.set(mvMatrix).invert().transpose();
	// upload matrices
	renderer->setProgramUniformFloatMatrix4x4(uniformMVPMatrix, mvpMatrix.getArray());
	renderer->setProgramUniformFloatMatrix4x4(uniformMVMatrix, mvMatrix.getArray());
	renderer->setProgramUniformFloatMatrix4x4(uniformNormalMatrix, normalMatrix.getArray());
}

void LightingShader::bindTexture(GLRenderer* renderer, int32_t textureId)
{
	// skip if not running
	if (isRunning == false)
		return;

	switch (renderer->getTextureUnit()) {
		case TEXTUREUNIT_DIFFUSE:
			renderer->setProgramUniformInteger(uniformDiffuseTextureAvailable, textureId == 0 ? 0 : 1);
			break;
		case TEXTUREUNIT_SPECULAR:
			if (renderer->isSpecularMappingAvailable() == false)
				break;

			renderer->setProgramUniformInteger(uniformSpecularTextureAvailable, textureId == 0 ? 0 : 1);
			break;
		case TEXTUREUNIT_NORMAL:
			if (renderer->isNormalMappingAvailable() == false)
				break;

			renderer->setProgramUniformInteger(uniformNormalTextureAvailable, textureId == 0 ? 0 : 1);
			break;
		case TEXTUREUNIT_DISPLACEMENT:
			if (renderer->isDisplacementMappingAvailable() == false)
				break;

			renderer->setProgramUniformInteger(uniformDisplacementTextureAvailable, textureId == 0 ? 0 : 1);
			break;
	}

}
