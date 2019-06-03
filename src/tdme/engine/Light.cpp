#include <tdme/engine/Light.h>

#include <tdme/engine/model/Color4.h>
#include <tdme/engine/subsystems/renderer/Renderer.h>
#include <tdme/math/Matrix4x4.h>
#include <tdme/math/Vector3.h>
#include <tdme/math/Vector4.h>

using tdme::engine::Light;
using tdme::engine::model::Color4;
using tdme::engine::subsystems::renderer::Renderer;
using tdme::math::Matrix4x4;
using tdme::math::Vector3;
using tdme::math::Vector4;

Light::Light()
{
	this->renderer = nullptr;
	this->id = -1;
	enabled = false;
	ambient.set(0.0f, 0.0f, 0.0f, 1.0f);
	diffuse.set(1.0f, 1.0f, 1.0f, 1.0f);
	specular.set(1.0f, 1.0f, 1.0f, 1.0f);
	position.set(0.0f, 0.0f, 0.0f, 0.0f);
	spotDirection.set(0.0f, 0.0f, -1.0f);
	spotExponent = 0.0f;
	spotCutOff = 180.0f;
	constantAttenuation = 1.0f;
	linearAttenuation = 0.0f;
	quadraticAttenuation = 0.0f;
}

Light::Light(Renderer* renderer, int32_t id) 
{
	this->renderer = renderer;
	this->id = id;
	enabled = false;
	ambient.set(0.0f, 0.0f, 0.0f, 1.0f);
	diffuse.set(1.0f, 1.0f, 1.0f, 1.0f);
	specular.set(1.0f, 1.0f, 1.0f, 1.0f);
	position.set(0.0f, 0.0f, 0.0f, 0.0f);
	spotDirection.set(0.0f, 0.0f, -1.0f);
	spotExponent = 0.0f;
	spotCutOff = 180.0f;
	constantAttenuation = 1.0f;
	linearAttenuation = 0.0f;
	quadraticAttenuation = 0.0f;
}

void Light::update(void* context) {
	if (enabled == true) {
		Vector4 lightPositionTransformed;
		Vector3 tmpVector3;
		Vector4 spotDirection4;
		Vector4 spotDirection4Transformed;
		renderer->setLightEnabled(context, id);
		renderer->setLightAmbient(context, id, ambient.getArray());
		renderer->setLightDiffuse(context, id, diffuse.getArray());
		if (renderer->isInstancedRenderingAvailable() == false) {
			renderer->setLightPosition(context, id, renderer->getCameraMatrix().multiply(position, lightPositionTransformed).scale(Math::abs(lightPositionTransformed.getW()) < Math::EPSILON?1.0f:1.0f / lightPositionTransformed.getW()).setW(1.0f).getArray());
			renderer->getCameraMatrix().multiply(spotDirection4.set(spotDirection, 0.0f), spotDirection4Transformed);
			renderer->setLightSpotDirection(context, id, tmpVector3.set(spotDirection4Transformed.getX(), spotDirection4Transformed.getY(), spotDirection4Transformed.getZ()).getArray());
		} else {
			// TODO: a.drewke, check if we can always use world space here or camera space
			renderer->setLightPosition(context, id, position.getArray());
			renderer->setLightSpotDirection(context, id, spotDirection.getArray());
		}
		renderer->setLightSpotExponent(context, id, spotExponent);
		renderer->setLightSpotCutOff(context, id, spotCutOff);
		renderer->setLightConstantAttenuation(context, id, constantAttenuation);
		renderer->setLightLinearAttenuation(context, id, linearAttenuation);
		renderer->setLightQuadraticAttenuation(context, id, quadraticAttenuation);
		renderer->onUpdateLight(context, id);
	} else {
		renderer->setLightDisabled(context, id);
		renderer->onUpdateLight(context, id);
	}
}
