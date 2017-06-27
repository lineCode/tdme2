// Generated from /tdme/src/tdme/engine/PartitionOctTree.java

#pragma once

#include <fwd-tdme.h>
#include <tdme/engine/fwd-tdme.h>
#include <tdme/engine/primitives/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/engine/Partition.h>

using tdme::engine::Partition;
using tdme::engine::Entity;
using tdme::engine::Frustum;
using tdme::engine::PartitionOctTree_PartitionTreeNode;
using tdme::engine::primitives::BoundingBox;
using tdme::engine::primitives::BoundingVolume;
using tdme::math::Vector3;
using tdme::utils::ArrayListIteratorMultiple;
using tdme::utils::Key;
using tdme::utils::Pool;
using tdme::utils::_ArrayList;
using tdme::utils::_HashMap;


struct default_init_tag;

/** 
 * Partition oct tree implementation
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::PartitionOctTree final
	: public Partition
{

public:
	typedef Partition super;

private:
	Key* key {  };
	ArrayListIteratorMultiple* entityIterator {  };
	BoundingBox* boundingBox {  };
	Vector3* halfExtension {  };
	Vector3* sideVector {  };
	Vector3* forwardVector {  };
	Vector3* upVector {  };
	Pool* entityPartitionNodesPool {  };
	Pool* boundingBoxPool {  };
	Pool* partitionTreeNodePool {  };
	Pool* subNodesPool {  };
	Pool* partitionEntitiesPool {  };
	Pool* keyPool {  };
	_HashMap* entityPartitionNodes {  };
	_ArrayList* visibleEntities {  };
	PartitionOctTree_PartitionTreeNode* treeRoot {  };

public:
	static constexpr float PARTITION_SIZE_MIN { 4.0f };
	static constexpr float PARTITION_SIZE_MID { 8.0f };
	static constexpr float PARTITION_SIZE_MAX { 16.0f };
protected:

	/** 
	 * Constructor
	 */
	void ctor();

public: /* protected */
	void reset() override;

public:

	/** 
	 * Creates a partition
	 * @param parent
	 * @param x
	 * @param y
	 * @param z
	 * @param partition size
	 * @return partition tree node
	 */
	PartitionOctTree_PartitionTreeNode* createPartition(PartitionOctTree_PartitionTreeNode* parent, int32_t x, int32_t y, int32_t z, float partitionSize);

public: /* protected */
	void addEntity(Entity* entity) override;
	void updateEntity(Entity* entity) override;
	void removeEntity(Entity* entity) override;

private:

	/** 
	 * Is partition empty
	 * @param node
	 * @return partition empty
	 */
	bool isPartitionNodeEmpty(PartitionOctTree_PartitionTreeNode* node);

	/** 
	 * Remove partition node, should be empty
	 * @param node
	 */
	void removePartitionNode(PartitionOctTree_PartitionTreeNode* node);

	/** 
	 * Do partition tree lookup
	 * @param frustum
	 * @param node
	 * @param visible entities
	 * @return number of look ups
	 */
	int32_t doPartitionTreeLookUpVisibleObjects(Frustum* frustum, PartitionOctTree_PartitionTreeNode* node, _ArrayList* visibleEntities);

public:
	_ArrayList* getVisibleEntities(Frustum* frustum) override;

private:

	/** 
	 * Do partition tree lookup
	 * @param node
	 * @param cbv
	 * @param cbvsIterator
	 */
	void addToPartitionTree(PartitionOctTree_PartitionTreeNode* node, Entity* entity, BoundingBox* cbv);

	/** 
	 * Add entity to tree
	 */
	void addToPartitionTree(Entity* entity, BoundingBox* cbv);

	/** 
	 * Do partition tree lookup for near entities to cbv
	 * @param node
	 * @param cbv
	 * @param entity iterator
	 */
	int32_t doPartitionTreeLookUpNearEntities(PartitionOctTree_PartitionTreeNode* node, BoundingBox* cbv, ArrayListIteratorMultiple* entityIterator);

public:
	ArrayListIteratorMultiple* getObjectsNearTo(BoundingVolume* cbv) override;
	ArrayListIteratorMultiple* getObjectsNearTo(Vector3* center) override;

	// Generated
	PartitionOctTree();
protected:
	PartitionOctTree(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();

private:
	void init();
	virtual ::java::lang::Class* getClass0();
	friend class PartitionOctTree_PartitionTreeNode;
	friend class PartitionOctTree_reset_1;
	friend class PartitionOctTree_reset_2;
	friend class PartitionOctTree_reset_3;
	friend class PartitionOctTree_reset_4;
	friend class PartitionOctTree_reset_5;
	friend class PartitionOctTree_reset_6;
};
