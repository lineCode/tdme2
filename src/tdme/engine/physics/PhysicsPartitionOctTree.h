#pragma once

#include <map>
#include <string>
#include <vector>

#include <tdme/tdme.h>
#include <tdme/engine/physics/fwd-tdme.h>
#include <tdme/engine/primitives/fwd-tdme.h>
#include <tdme/engine/primitives/BoundingBox.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/math/Vector3.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/engine/physics/PhysicsPartition.h>
#include <tdme/engine/physics/PhysicsPartitionOctTree_PartitionTreeNode.h>
#include <tdme/engine/physics/RigidBody.h>
#include <tdme/utils/VectorIteratorMultiple.h>
#include <tdme/utils/Pool.h>

using std::map;
using std::vector;
using std::string;

using tdme::engine::physics::PhysicsPartition;
using tdme::engine::physics::PhysicsPartitionOctTree_PartitionTreeNode;
using tdme::engine::physics::RigidBody;
using tdme::engine::primitives::BoundingBox;
using tdme::engine::primitives::BoundingVolume;
using tdme::math::Vector3;
using tdme::utils::VectorIteratorMultiple;
using tdme::utils::Pool;

/** 
 * Partition oct tree implementation
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::physics::PhysicsPartitionOctTree final
	: public PhysicsPartition
{
	friend class PhysicsPartitionOctTree_PartitionTreeNode;

private:
	static constexpr float PARTITION_SIZE_MIN { 16.0f };
	static constexpr float PARTITION_SIZE_MAX { 64.0f };

	VectorIteratorMultiple<RigidBody*> rigidBodyIterator {  };
	BoundingBox boundingBox {  };
	Vector3 halfExtension {  };
	Vector3 sideVector {  };
	Vector3 forwardVector {  };
	Vector3 upVector {  };
	map<string, vector<PhysicsPartitionOctTree_PartitionTreeNode*>> rigidBodyPartitionNodes {  };
	PhysicsPartitionOctTree_PartitionTreeNode treeRoot {  };

	/** 
	 * Reset
	 */
	void reset() override;

	/** 
	 * Is partition empty
	 * @param node
	 * @return partition empty
	 */
	bool isPartitionNodeEmpty(PhysicsPartitionOctTree_PartitionTreeNode* node);

	/** 
	 * Remove partition node, should be empty
	 * @param node
	 */
	void removePartitionNode(PhysicsPartitionOctTree_PartitionTreeNode* node);

	/** 
	 * Do partition tree lookup
	 * @param node
	 * @param cbv
	 * @param cbvsIterator
	 */
	void addToPartitionTree(PhysicsPartitionOctTree_PartitionTreeNode* node, RigidBody* rigidBody, BoundingBox* cbv);

	/** 
	 * Add rigidBody to tree
	 */
	void addToPartitionTree(RigidBody* rigidBody, BoundingBox* cbv);

	/** 
	 * Do partition tree lookup for near entities to cbv
	 * @param node
	 * @param cbv
	 * @param rigidBody iterator
	 */
	int32_t doPartitionTreeLookUpNearEntities(PhysicsPartitionOctTree_PartitionTreeNode* node, BoundingBox* cbv, VectorIteratorMultiple<RigidBody*>& rigidBodyIterator);

	/**
	 * Adds a object
	 * @param rigidBody
	 */
	void addRigidBody(RigidBody* rigidBody) override;

	/**
	 * Updates a object
	 * @param rigidBody
	 */
	void updateRigidBody(RigidBody* rigidBody) override;

	/**
	 * Removes a rigid body
	 * @param rigid body
	 */
	void removeRigidBody(RigidBody* rigidBody) override;

public:

	/**
	 * Creates a partition
	 * @param parent
	 * @param x
	 * @param z
	 * @param partition size
	 * @return partition tree node
	 */
	PhysicsPartitionOctTree_PartitionTreeNode* createPartition(PhysicsPartitionOctTree_PartitionTreeNode* parent, int32_t x, int32_t y, int32_t z, float partitionSize);

	/**
	 * Get objects near to rigid body
	 * @param cbv
	 * @return objects near to cbv
	 */
	VectorIteratorMultiple<RigidBody*>* getObjectsNearTo(RigidBody* rigidBody) override;

	/**
	 * Get objects near to
	 * @param cbv
	 * @return objects near to cbv
	 */
	VectorIteratorMultiple<RigidBody*>* getObjectsNearTo(BoundingVolume* cbv) override;

	/** 
	 * Get objects near to
	 * @param cbv
	 * @return objects near to cbv
	 */
	VectorIteratorMultiple<RigidBody*>* getObjectsNearTo(const Vector3& center) override;

	/**
	 * Public constructor
	 */
	PhysicsPartitionOctTree();
};
