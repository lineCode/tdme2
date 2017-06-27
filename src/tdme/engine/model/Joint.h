// Generated from /tdme/src/tdme/engine/model/Joint.java

#pragma once

#include <java/lang/fwd-tdme.h>
#include <tdme/engine/model/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <java/lang/Object.h>

using java::lang::Object;
using java::lang::String;
using tdme::math::Matrix4x4;


struct default_init_tag;

/** 
 * Joint / Bone
 * @author andreas.drewke
 */
class tdme::engine::model::Joint final
	: public Object
{

public:
	typedef Object super;

private:
	String* groupId {  };
	Matrix4x4* bindMatrix {  };
protected:

	/** 
	 * Public constructor
	 * @param group id
	 * @param bind matrix
	 */
	void ctor(String* groupId);

public:

	/** 
	 * Associated group or bone id
	 * @return group id
	 */
	String* getGroupId();

	/** 
	 * Bind matrix
	 * @return matrix
	 */
	Matrix4x4* getBindMatrix();

	/** 
	 * @return string representation
	 */
	String* toString() override;

	// Generated
	Joint(String* groupId);
protected:
	Joint(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();

private:
	virtual ::java::lang::Class* getClass0();
};
