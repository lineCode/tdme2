#pragma once

#include <tdme/tdme.h>
#include <tdme/engine/fwd-tdme.h>
#include <tdme/engine/subsystems/lines/fwd-tdme.h>
#include <tdme/engine/subsystems/renderer/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/math/Matrix4x4.h>

using tdme::engine::Engine;
using tdme::engine::subsystems::renderer::Renderer;
using tdme::math::Matrix4x4;

/** 
 * Lines shader program
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::subsystems::lines::LinesShader final
{
private:
	int32_t renderProgramId {  };
	int32_t renderFragmentShaderId {  };
	int32_t renderVertexShaderId {  };
	int32_t uniformMVPMatrix {  };
	int32_t uniformDiffuseTextureUnit {  };
	int32_t uniformEffectColorMul {  };
	int32_t uniformEffectColorAdd {  };
	Matrix4x4 mvpMatrix {  };
	bool isRunning {  };
	bool initialized {  };
	Engine* engine {  };
	Renderer* renderer {  };

public:

	/** 
	 * @return initialized and ready to be used
	 */
	bool isInitialized();

	/** 
	 * Initialize renderer
	 */
	void initialize();

	/** 
	 * Use lighting program
	 * @param context context
	 */
	void useProgram(void* context);

	/** 
	 * Update effect to program
	 * @param renderer renderer
	 * @param context context
	 */
	void updateEffect(Renderer* renderer, void* context);

	/** 
	 * Unuse particles shader program
	 * @param context context
	 */
	void unUseProgram(void* context);

	/** 
	 * Update matrices to program
	 * @param renderer renderer
	 * @param context context
	 */
	void updateMatrices(Renderer* renderer, void* context);

	/**
	 * Set parameters
	 * @param context context
	 * @param textureId texture id
	 * @param lineWidth line width
	 */
	void setParameters(void* context, int32_t textureId, float lineWidth);

	/**
	 * Public constructor
	 * @param engine engine
	 * @param renderer renderer
	 */
	LinesShader(Engine* engine, Renderer* renderer);
};
