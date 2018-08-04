#pragma once

#include <tdme/tdme.h>
#include <tdme/engine/subsystems/renderer/fwd-tdme.h>
#include <tdme/engine/subsystems/shadowmapping/fwd-tdme.h>
#include <tdme/math/Matrix4x4.h>

using tdme::engine::subsystems::renderer::GLRenderer;
using tdme::engine::subsystems::shadowmapping::ShadowMappingShaderPreImplementation;
using tdme::math::Matrix4x4;

/** 
 * Pre shadow mapping shader for render shadow map pass 
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::subsystems::shadowmapping::ShadowMappingShaderPre final
{
private:
	ShadowMappingShaderPreImplementation* defaultImplementation { nullptr };
	ShadowMappingShaderPreImplementation* foliageImplementation { nullptr };
	ShadowMappingShaderPreImplementation* implementation { nullptr };
	bool running { false };

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
	 */
	void useProgram();

	/** 
	 * Un use pre render shadow mapping program
	 */
	void unUseProgram();

	/** 
	 * Set up pre program mvp matrix
	 * @param mvp matrix
	 */
	void updateMatrices(const Matrix4x4& mvpMatrix);

	/**
	 * Set up pre program texture matrix
	 * @param renderer
	 */
	void updateTextureMatrix(GLRenderer* renderer);

	/**
	 * Update material
	 * @param renderer
	 */
	void updateMaterial(GLRenderer* renderer);

	/**
	 * Bind texture
	 * @param renderer
	 * @param texture id
	 */
	void bindTexture(GLRenderer* renderer, int32_t textureId);

	/**
	 * Update apply foliage animation
	 * @param renderer
	 */
	void updateApplyFoliageAnimation(GLRenderer* renderer);

	/**
	 * Constructor
	 * @param renderer
	 */
	ShadowMappingShaderPre(GLRenderer* renderer);

	/**
	 * Destructor
	 */
	~ShadowMappingShaderPre();

};
