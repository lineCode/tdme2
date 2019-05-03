#include <tdme/engine/subsystems/shadowmapping/ShadowMappingShaderRenderDefaultImplementation.h>

#include <string>

#include <tdme/engine/subsystems/renderer/GLRenderer.h>
#include <tdme/os/filesystem/FileSystem.h>
#include <tdme/os/filesystem/FileSystemInterface.h>

using std::to_string;

using tdme::engine::subsystems::shadowmapping::ShadowMappingShaderRenderDefaultImplementation;
using tdme::engine::subsystems::renderer::GLRenderer;
using tdme::os::filesystem::FileSystem;
using tdme::os::filesystem::FileSystemInterface;

bool ShadowMappingShaderRenderDefaultImplementation::isSupported(GLRenderer* renderer) {
	return true;
}

ShadowMappingShaderRenderDefaultImplementation::ShadowMappingShaderRenderDefaultImplementation(GLRenderer* renderer): ShadowMappingShaderRenderBaseImplementation(renderer)
{
}

ShadowMappingShaderRenderDefaultImplementation::~ShadowMappingShaderRenderDefaultImplementation()
{
}

void ShadowMappingShaderRenderDefaultImplementation::initialize()
{
	auto rendererVersion = renderer->getGLVersion();

	// load shadow mapping shaders
	renderVertexShaderId = renderer->loadShader(
		renderer->SHADER_VERTEX_SHADER,
		"shader/" + rendererVersion + "/shadowmapping",
		"render_vertexshader.c",
		"",
		FileSystem::getInstance()->getContentAsString(
			"shader/" + rendererVersion + "/shadowmapping",
			"render_computevertex.inc.c"
		)
	);
	if (renderVertexShaderId == 0) return;

	renderFragmentShaderId = renderer->loadShader(
		renderer->SHADER_FRAGMENT_SHADER,
		"shader/" + rendererVersion + "/shadowmapping",
		"render_fragmentshader.c"
	);
	if (renderFragmentShaderId == 0) return;

	// create shadow mapping render program
	renderProgramId = renderer->createProgram();
	renderer->attachShaderToProgram(renderProgramId, renderVertexShaderId);
	renderer->attachShaderToProgram(renderProgramId, renderFragmentShaderId);

	ShadowMappingShaderRenderBaseImplementation::initialize();
}
