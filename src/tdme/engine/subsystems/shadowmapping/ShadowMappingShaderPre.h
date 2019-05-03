#pragma once

#include <map>
#include <string>

#include <tdme/tdme.h>
#include <tdme/engine/fwd-tdme.h>
#include <tdme/engine/subsystems/renderer/fwd-tdme.h>
#include <tdme/engine/subsystems/shadowmapping/fwd-tdme.h>
#include <tdme/math/Matrix4x4.h>

using std::map;
using std::string;

using tdme::engine::Engine;
using tdme::engine::subsystems::renderer::Renderer;
using tdme::engine::subsystems::shadowmapping::ShadowMappingShaderPreImplementation;
using tdme::math::Matrix4x4;

/** 
 * Shadow mapping shader to create a shadow map
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::subsystems::shadowmapping::ShadowMappingShaderPre final
{
private:
	map<string, ShadowMappingShaderPreImplementation*> shader;
	ShadowMappingShaderPreImplementation* implementation { nullptr };
	bool running { false };
	Engine* engine { nullptr };

public:

	/** 
	 * @return if initialized and ready to use
	 */
	bool isInitialized();

	/** 
	 * Init shadow mapping
	 */
	void initialize();

	/** 
	 * Use pre render shadow mapping program
	 * @param engine engine
	 */
	void useProgram(Engine* engine);

	/** 
	 * Un use pre render shadow mapping program
	 */
	void unUseProgram();

	/** 
	 * Set up pre program mvp matrix
	 * @param mvpMatrix mvp matrix
	 */
	void updateMatrices(const Matrix4x4& mvpMatrix);

	/**
	 * Set up pre program texture matrix
	 * @param renderer renderer
	 */
	void updateTextureMatrix(Renderer* renderer);

	/**
	 * Update material
	 * @param renderer renderer
	 */
	void updateMaterial(Renderer* renderer);

	/**
	 * Bind texture
	 * @param renderer renderer
	 * @param textureId texture id
	 */
	void bindTexture(Renderer* renderer, int32_t textureId);

	/**
	 * Set shader
	 * @param id shader id
	 */
	void setShader(const string& id);

	/**
	 * Constructor
	 * @param renderer renderer
	 */
	ShadowMappingShaderPre(Renderer* renderer);

	/**
	 * Destructor
	 */
	~ShadowMappingShaderPre();

};
