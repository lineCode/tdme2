#pragma once

#include <array>
#include <vector>

#include <ext/reactphysics3d/src/collision/shapes/ConvexMeshShape.h>

#include <tdme/math/Vector3.h>
#include <tdme/engine/primitives/fwd-tdme.h>
#include <tdme/engine/primitives/ConvexMeshBoundingVolume.h>

using std::array;
using std::vector;

using tdme::engine::primitives::BoundingVolume;
using tdme::engine::primitives::BoundingBox;
using tdme::math::Vector3;

/** 
 * Oriented Bounding Box
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::primitives::OrientedBoundingBox final
	: public ConvexMeshBoundingVolume
{
private:
	static const array<int32_t, 3> FACE0_INDICES;
	static const array<int32_t, 3> FACE1_INDICES;
	static const array<int32_t, 3> FACE2_INDICES;
	static const array<int32_t, 3> FACE3_INDICES;
	static const array<int32_t, 3> FACE4_INDICES;
	static const array<int32_t, 3> FACE5_INDICES;
	static const array<int32_t, 3> FACE6_INDICES;
	static const array<int32_t, 3> FACE7_INDICES;
	static const array<int32_t, 3> FACE8_INDICES;
	static const array<int32_t, 3> FACE9_INDICES;
	static const array<int32_t, 3> FACE10_INDICES;
	static const array<int32_t, 3> FACE11_INDICES;
	static const array<array<int32_t,3>,12> facesVerticesIndexes;

	/**
	 * Create vertices
	 */
	void createVertices();

	/**
	 * Create convex mesh
	 */
	void createConvexMesh();
public:
	static const Vector3 AABB_AXIS_X;
	static const Vector3 AABB_AXIS_Y;
	static const Vector3 AABB_AXIS_Z;

public:
	/** 
	 * @return center
	 */
	const Vector3& getCenter() const;

	/**
	 * @return 3 axes
	 */
	const array<Vector3, 3>* getAxes() const;

	/** 
	 * @return half extension
	 */
	const Vector3& getHalfExtension() const;

	/** 
	 * Set up oriented bounding box from oriented bounding box
	 * @param bb
	 */

	// overrides
	BoundingVolume* clone() const override;
	/** 
	 * @return oriented bounding box vertices
	 */
	inline const vector<Vector3>* getVertices() const {
		return &vertices;
	}

	/** 
	 * @return faces vertices indexes
	 */
	inline static const array<array<int32_t,3>,12>* getFacesVerticesIndexes() {
		return &facesVerticesIndexes;
	}

	/**
	 * Public constructor
	 * @param center
	 * @param axis0
	 * @param axis1
	 * @param axis2
	 * @param half extension
	 */
	OrientedBoundingBox(const Vector3& center, const Vector3& axis0, const Vector3& axis1, const Vector3& axis2, const Vector3& halfExtension);

	/**
	 * Public constructor
	 * @param bounding box
	 */
	OrientedBoundingBox(BoundingBox* bb);

	/**
	 * Public constructor
	 */
	OrientedBoundingBox();

private:
	Vector3 center {  };
	array<Vector3, 3> axes {  };
	Vector3 halfExtension {  };
	vector<Vector3> vertices {  };
};
