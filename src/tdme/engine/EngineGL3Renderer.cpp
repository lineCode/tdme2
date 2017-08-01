// Generated from /tdme/src/tdme/engine/Engine.java
#include <tdme/engine/EngineGL3Renderer.h>

#include <tdme/engine/Engine.h>
#include <tdme/engine/subsystems/lighting/LightingShader.h>
#include <tdme/engine/subsystems/particlesystem/ParticlesShader.h>
#include <tdme/engine/subsystems/shadowmapping/ShadowMapping.h>
#include <tdme/gui/renderer/GUIShader.h>

using tdme::engine::EngineGL3Renderer;
using tdme::engine::Engine;
using tdme::engine::subsystems::lighting::LightingShader;
using tdme::engine::subsystems::particlesystem::ParticlesShader;
using tdme::engine::subsystems::shadowmapping::ShadowMapping;
using tdme::gui::renderer::GUIShader;

EngineGL3Renderer::EngineGL3Renderer(Engine* engine) :
	engine(engine)
{
}

void EngineGL3Renderer::onUpdateProjectionMatrix()
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateMatrices(this);

	if (Engine::particlesShader != nullptr)
		Engine::particlesShader->updateMatrices(this);

	if (engine->shadowMapping != nullptr)
		engine->shadowMapping->updateMVPMatrices(this);

}

void EngineGL3Renderer::onUpdateCameraMatrix()
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateMatrices(this);

	if (Engine::particlesShader != nullptr)
		Engine::particlesShader->updateMatrices(this);

	if (engine->shadowMapping != nullptr)
		engine->shadowMapping->updateMVPMatrices(this);

}

void EngineGL3Renderer::onUpdateModelViewMatrix()
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateMatrices(this);

	if (Engine::particlesShader != nullptr)
		Engine::particlesShader->updateMatrices(this);

	if (engine->shadowMapping != nullptr)
		engine->shadowMapping->updateMVPMatrices(this);

}

void EngineGL3Renderer::onBindTexture(int32_t textureId)
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->bindTexture(this, textureId);

	if (Engine::guiShader != nullptr)
		Engine::guiShader->bindTexture(this, textureId);

}

void EngineGL3Renderer::onUpdateTextureMatrix()
{
}

void EngineGL3Renderer::onUpdateEffect()
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateEffect(this);

	if (Engine::particlesShader != nullptr)
		Engine::particlesShader->updateEffect(this);

	if (Engine::guiShader != nullptr)
		Engine::guiShader->updateEffect(this);

}

void EngineGL3Renderer::onUpdateLight(int32_t lightId)
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateLight(this, lightId);

}

void EngineGL3Renderer::onUpdateMaterial()
{
	if (Engine::lightingShader != nullptr)
		Engine::lightingShader->updateMaterial(this);

}
