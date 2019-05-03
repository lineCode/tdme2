#include <string>

#include <tdme/engine/Engine.h>
#include <tdme/engine/subsystems/postprocessing/PostProcessingShaderBaseImplementation.h>
#include <tdme/engine/subsystems/postprocessing/PostProcessingShaderSSAOImplementation.h>
#include <tdme/engine/subsystems/renderer/Renderer.h>

using std::string;

using tdme::engine::subsystems::postprocessing::PostProcessingShaderSSAOImplementation;
using tdme::engine::subsystems::renderer::Renderer;
using tdme::engine::Engine;

bool PostProcessingShaderSSAOImplementation::isSupported(Renderer* renderer) {
	return renderer->getGLVersion() == "gl3";
}

PostProcessingShaderSSAOImplementation::PostProcessingShaderSSAOImplementation(Renderer* renderer): PostProcessingShaderBaseImplementation(renderer)
{
}

void PostProcessingShaderSSAOImplementation::initialize()
{
	auto rendererVersion = renderer->getGLVersion();

	//	fragment shader
	fragmentShaderId = renderer->loadShader(
		renderer->SHADER_FRAGMENT_SHADER,
		"shader/" + rendererVersion + "/postprocessing",
		"ssao_fragmentshader.c",
		Engine::is4K() == true?"#define HAVE_4K":""
	);
	if (fragmentShaderId == 0) return;

	//	vertex shader
	vertexShaderId = renderer->loadShader(
		renderer->SHADER_VERTEX_SHADER,
		"shader/" + rendererVersion + "/postprocessing",
		"ssao_vertexshader.c"
	);
	if (vertexShaderId == 0) return;

	// create, attach and link program
	programId = renderer->createProgram();
	renderer->attachShaderToProgram(programId, vertexShaderId);
	renderer->attachShaderToProgram(programId, fragmentShaderId);

	//
	PostProcessingShaderBaseImplementation::initialize();
}
