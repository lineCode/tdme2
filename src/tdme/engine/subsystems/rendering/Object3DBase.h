#pragma once

#include <map>
#include <string>
#include <vector>

#include <tdme/tdme.h>
#include <tdme/engine/fwd-tdme.h>
#include <tdme/engine/Engine.h>
#include <tdme/engine/model/fwd-tdme.h>
#include <tdme/engine/primitives/fwd-tdme.h>
#include <tdme/engine/subsystems/rendering/fwd-tdme.h>
#include <tdme/engine/subsystems/rendering/AnimationState.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/engine/Transformations.h>

using std::map;
using std::vector;
using std::string;

using tdme::engine::Engine;
using tdme::engine::Transformations;
using tdme::engine::model::Group;
using tdme::engine::model::Model;
using tdme::engine::primitives::BoundingVolume;
using tdme::engine::primitives::Triangle;
using tdme::engine::subsystems::rendering::AnimationState;
using tdme::engine::subsystems::rendering::Object3DBase_TransformedFacesIterator;
using tdme::engine::subsystems::rendering::Object3DGroup;
using tdme::engine::subsystems::rendering::Object3DGroupMesh;
using tdme::math::Matrix4x4;

/** 
 * Object3D base class
 * @author Andreas Drewke
 */
class tdme::engine::subsystems::rendering::Object3DBase
	: public Transformations
{
	friend class Object3DGroup;
	friend class Object3DBase_TransformedFacesIterator;
	friend class ModelUtilitiesInternal;

private:
	Object3DBase_TransformedFacesIterator* transformedFacesIterator { nullptr };

	/**
	 * Determine skinned group count
	 * @param groups groups
	 */
	int32_t determineSkinnedGroupCount(const map<string, Group*>& groups);

	/**
	 * Determine skinned group count
	 * @param map* groups
	 * @param count current count
	 */
	int32_t determineSkinnedGroupCount(const map<string, Group*>&, int32_t count);

	/**
	 * Determine skinned groups
	 * @param map* groups
	 * @param skinningGroups skinning groups
	 * @param idx idx
	 */
	int32_t determineSkinnedGroups(const map<string, Group*>&, vector<Group*>& skinningGroups, int32_t idx);


protected:
	Model* model;
	map<string, Matrix4x4*> overridenTransformationsMatrices;
	vector<map<string, Matrix4x4*>> transformationsMatrices;
	bool hasSkinning;
	bool hasAnimations;
	vector<map<string, Matrix4x4*>> skinningGroupsMatrices;
	vector<Group*> skinningGroups;
	vector<AnimationState> baseAnimations;
	int baseAnimationIdx;
	map<string, AnimationState*> overlayAnimationsById;
	map<string, AnimationState*> overlayAnimationsByJointId;
	vector<Object3DGroup*> object3dGroups;
	bool usesManagers;
	Engine::AnimationProcessingTarget animationProcessingTarget;

	/**
	 * Creates all groups transformation matrices
	 * @param matrices matrices
	 * @param groups groups
	 */
	virtual void createTransformationsMatrices(map<string, Matrix4x4*>& matrices, const map<string, Group*>& groups);

	/**
	 * Calculates all groups transformation matrices
	 * @param groups groups
	 * @param parentTransformationsMatrix parent transformations matrix
	 * @param animationState animation state
	 * @param transformationsMatrices transformations matrices which need to be set up
	 * @param depth depth
	 */
	virtual void computeTransformationsMatrices(const map<string, Group*>& groups, const Matrix4x4 parentTransformationsMatrix, AnimationState* animationState, map<string, Matrix4x4*>& transformationsMatrices, int32_t depth);

	/**
	 * Compute transformations for given animation state into given transformations matrices
	 * @param baseAnimation base animation
	 * @param transformationsMatrices transformations matrices
	 * @param context context
	 * @param lastFrameAtTime time of last animation computation
	 * @param currentFrameAtTime time of current animation computation
	 */
	virtual void computeTransformations(AnimationState& baseAnimation, map<string, Matrix4x4*>& transformationsMatrices, void* context, int64_t lastFrameAtTime, int64_t currentFrameAtTime);

	/**
	 * Update skinning transformations matrices
	 * @param transformationsMatrices transformations matrices
	 */
	virtual void updateSkinningTransformationsMatrices(const map<string, Matrix4x4*>& transformationsMatrices);

	/**
	 * Get skinning groups matrices
	 * @param group group
	 * @return matrices
	 */
	virtual map<string, Matrix4x4*>* getSkinningGroupsMatrices(Group* group);

	/**
	 * Public constructor
	 * @param model model
	 * @param useManagers use mesh and object 3d group renderer model manager
	 * @param animationProcessingTarget animation processing target
	 */
	Object3DBase(Model* model, bool useManagers, Engine::AnimationProcessingTarget animationProcessingTarget);

	/**
	 * Destructor
	 */
	virtual ~Object3DBase();

public:

	/** 
	 * @return model
	 */
	inline virtual Model* getModel() {
		return model;
	}

	/** 
	 * Sets up a base animation to play
	 * @param id id
	 * @param speed speed whereas 1.0 is default speed
	 */
	virtual void setAnimation(const string& id, float speed = 1.0f);

	/**
	 * Set up animation speed
	 * @param speed speed whereas 1.0 is default speed
	 */
	virtual void setAnimationSpeed(float speed);

	/** 
	 * Overlays a animation above the base animation
	 * @param id id
	 */
	virtual void addOverlayAnimation(const string& id);

	/** 
	 * Removes a overlay animation
	 * @param id id
	 */
	virtual void removeOverlayAnimation(const string& id);

	/** 
	 * Removes all finished overlay animations
	 */
	virtual void removeOverlayAnimationsFinished();

	/** 
	 * Removes all overlay animations
	 */
	virtual void removeOverlayAnimations();

	/** 
	 * @return active animation setup id
	 */
	virtual const string getAnimation();

	/** 
	 * Returns current base animation time 
	 * @return 0.0 <= time <= 1.0
	 */
	virtual float getAnimationTime();

	/** 
	 * Returns if there is currently running a overlay animation with given id
	 * @param id id
	 * @return animation is running
	 */
	virtual bool hasOverlayAnimation(const string& id);

	/** 
	 * Returns current overlay animation time
	 * @param id id 
	 * @return 0.0 <= time <= 1.0
	 */
	virtual float getOverlayAnimationTime(const string& id);

	/** 
	 * Returns transformation matrix for given group
	 * @param id group id
	 * @return transformation matrix or identity matrix if not found
	 */
	virtual const Matrix4x4 getTransformationsMatrix(const string& id);

	/**
	 * Set transformation matrix for given group
	 * @param id group id
	 * @param matrix transformation matrix
	 */
	virtual void setTransformationsMatrix(const string& id, const Matrix4x4& matrix);

	/**
	 * Unset transformation matrix for given group
	 * @param id group id
	 */
	virtual void unsetTransformationsMatrix(const string& id);

	/**
	 * Pre render step, computes transformations
	 * @param context context
	 * @param lastFrameAtTime time of last animation computation
	 * @param currentFrameAtTime time of current animation computation
	 */
	virtual void computeTransformations(void* context, int64_t lastFrameAtTime, int64_t currentFrameAtTime);

	/**
	 * @return group count
	 */
	virtual int getGroupCount() const;

	/** 
	 * Retrieves list of triangles of all or given groups
	 * @param triangles triangles
	 * @param groupIdx group index or -1 for all groups
	 */
	virtual void getTriangles(vector<Triangle>& triangles, int groupIdx = -1);

	/** 
	 * @return transformed faces iterator
	 */
	virtual Object3DBase_TransformedFacesIterator* getTransformedFacesIterator();

	/** 
	 * Returns object3d group mesh object
	 * @param groupId group id
	 * @return object3d group mesh object
	 */
	virtual Object3DGroupMesh* getMesh(const string& groupId);

	/** 
	 * Initiates this object3d 
	 */
	virtual void initialize();

	/** 
	 * Disposes this object3d 
	 */
	virtual void dispose();

	// overriden methods
	virtual const Matrix4x4& getTransformationsMatrix() const;
};
