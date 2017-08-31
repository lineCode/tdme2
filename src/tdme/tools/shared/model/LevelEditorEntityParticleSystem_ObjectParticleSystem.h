// Generated from /tdme/src/tdme/tools/shared/model/LevelEditorEntityParticleSystem.java

#pragma once

#include <fwd-tdme.h>
#include <java/lang/fwd-tdme.h>
#include <tdme/engine/model/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/math/Vector3.h>
#include <tdme/tools/shared/model/fwd-tdme.h>

using java::lang::String;
using tdme::engine::model::Model;
using tdme::math::Vector3;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_BoundingBoxParticleEmitter;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_CircleParticleEmitter;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_CircleParticleEmitterPlaneVelocity;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_Emitter;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_PointParticleEmitter;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_PointParticleSystem;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_SphereParticleEmitter;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem_Type;
using tdme::tools::shared::model::LevelEditorEntityParticleSystem;

/** 
 * Object particle system
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::tools::shared::model::LevelEditorEntityParticleSystem_ObjectParticleSystem
{
	friend class LevelEditorEntityParticleSystem;
	friend class LevelEditorEntityParticleSystem_Type;
	friend class LevelEditorEntityParticleSystem_PointParticleSystem;
	friend class LevelEditorEntityParticleSystem_Emitter;
	friend class LevelEditorEntityParticleSystem_PointParticleEmitter;
	friend class LevelEditorEntityParticleSystem_BoundingBoxParticleEmitter;
	friend class LevelEditorEntityParticleSystem_CircleParticleEmitter;
	friend class LevelEditorEntityParticleSystem_CircleParticleEmitterPlaneVelocity;
	friend class LevelEditorEntityParticleSystem_SphereParticleEmitter;

private:
	Vector3 scale {  };
	int32_t maxCount {  };
	bool autoEmit {  };
	Model* model {  };
	String* modelFileName {  };

public:

	/** 
	 * @return scale
	 */
	virtual Vector3* getScale();

	/** 
	 * @return max count
	 */
	virtual int32_t getMaxCount();

	/** 
	 * Set max count
	 * @param max count
	 */
	virtual void setMaxCount(int32_t maxCount);

	/** 
	 * @return is auto emit
	 */
	virtual bool isAutoEmit();

	/** 
	 * Set auto emit 
	 * @param autoEmit
	 */
	virtual void setAutoEmit(bool autoEmit);

	/** 
	 * @return model
	 */
	virtual Model* getModel();

	/** 
	 * Set model
	 * @param model
	 */
	virtual void setModel(Model* model);

	/** 
	 * @return model file
	 */
	virtual String* getModelFile();

	/** 
	 * Set model file
	 * @param model file name
	 */
	virtual void setModelFile(String* modelFileName) /* throws(Exception) */;

	/**
	 * Public constructor
	 */
	LevelEditorEntityParticleSystem_ObjectParticleSystem();
};
