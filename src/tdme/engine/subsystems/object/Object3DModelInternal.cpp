// Generated from /tdme/src/tdme/engine/subsystems/object/Object3DModelInternal.java
#include <tdme/engine/subsystems/object/Object3DModelInternal.h>

using tdme::engine::subsystems::object::Object3DModelInternal;

Object3DModelInternal::Object3DModelInternal(Model* model) :
	Object3DBase(model, false, Engine::AnimationProcessingTarget::CPU_NORENDERING)
{
}

