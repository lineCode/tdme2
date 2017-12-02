#include <tdme/engine/physics/CollisionDetection.h>

#include <string>

#include <tdme/math/Math.h>
#include <tdme/engine/physics/CollisionResponse_Entity.h>
#include <tdme/engine/physics/CollisionResponse.h>
#include <tdme/engine/primitives/BoundingBox.h>
#include <tdme/engine/primitives/BoundingVolume.h>
#include <tdme/engine/primitives/Capsule.h>
#include <tdme/engine/primitives/ConvexMesh.h>
#include <tdme/engine/primitives/LineSegment.h>
#include <tdme/engine/primitives/OrientedBoundingBox.h>
#include <tdme/engine/primitives/Sphere.h>
#include <tdme/engine/primitives/SphereInternal.h>
#include <tdme/engine/primitives/Triangle.h>
#include <tdme/math/MathTools.h>
#include <tdme/math/SeparatingAxisTheorem.h>
#include <tdme/math/TriangleTriangleIntersection.h>
#include <tdme/math/Vector3.h>
#include <tdme/utils/Float.h>
#include <tdme/utils/Console.h>

using std::string;

using tdme::engine::physics::CollisionDetection;
using tdme::math::Math;
using tdme::engine::physics::CollisionResponse_Entity;
using tdme::engine::physics::CollisionResponse;
using tdme::engine::primitives::BoundingBox;
using tdme::engine::primitives::BoundingVolume;
using tdme::engine::primitives::Capsule;
using tdme::engine::primitives::ConvexMesh;
using tdme::engine::primitives::LineSegment;
using tdme::engine::primitives::OrientedBoundingBox;
using tdme::engine::primitives::Sphere;
using tdme::engine::primitives::SphereInternal;
using tdme::engine::primitives::Triangle;
using tdme::math::MathTools;
using tdme::math::SeparatingAxisTheorem;
using tdme::math::TriangleTriangleIntersection;
using tdme::math::Vector3;
using tdme::utils::Float;
using tdme::utils::Console;

constexpr bool CollisionDetection::VERBOSE;

constexpr bool CollisionDetection::CHECK_COLLISIONRESPONSE;

constexpr array<int32_t,6> CollisionDetection::LINESEGMENTSTRIANGLEINDICES;

bool CollisionDetection::doCollide(SphereInternal* s1, SphereInternal* s2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	{
		Vector3 axis;
		if (axis.set(s1->getCenter()).sub(s2->getCenter()).computeLengthSquared() <= (s1->getSphereRadius() + s2->getSphereRadius()) * (s1->getSphereRadius() + s2->getSphereRadius()) == false) return false;
	}

	Vector3 axis;
	Vector3 hitPoint;
	axis.set(s2->getCenter()).sub(s1->getCenter());
	if (checkMovementFallback(axis, movement, collision) == true) {
		auto collisionEntity = collision->getEntityAt(0);
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(s1->getRadius()).add(s1->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(-s2->getRadius()).add(s2->getCenter()));
		return true;
	}
	auto distance = axis.computeLength();
	auto _distance = distance - (s1->getRadius() + s2->getRadius());
	if (_distance < 0.0f) {
		auto collisionEntity = collision->addResponse(_distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(s1->getRadius()).add(s1->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-s2->getRadius()).add(s2->getCenter()));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, SphereInternal* sphere, const Vector3& movement, CollisionResponse* collision)
{
	{
		Vector3 axis;
		if (axis.set(obb->getCenter()).sub(sphere->getCenter()).computeLengthSquared() <= (obb->getSphereRadius() + sphere->getSphereRadius()) * (obb->getSphereRadius() + sphere->getSphereRadius()) == false) return false;
	}

	collision->reset();

	Vector3 closestPoint;
	Vector3 axis;
	auto& sphereCenter = sphere->getCenter();
	obb->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(sphereCenter).sub(closestPoint);
	float distance;
	if (axis.computeLength() < MathTools::EPSILON) {
		Vector3 pointOnFaceNearest;
		Vector3 pointOnFaceOpposite;
		obb->computeNearestPointOnFaceBoundingVolume(closestPoint, pointOnFaceNearest);
		obb->computeOppositePointOnFaceBoundingVolume(closestPoint, pointOnFaceOpposite);
		obb->computeNearestPointOnFaceBoundingVolume(pointOnFaceNearest, closestPoint);
		axis.set(sphereCenter).sub(pointOnFaceNearest);
		distance = -axis.computeLength() - sphere->getRadius();
		axis.set(pointOnFaceNearest).sub(pointOnFaceOpposite);
	} else {
		distance = axis.computeLength() - sphere->getRadius();
	}
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(closestPoint);
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-sphere->getRadius()).add(sphereCenter));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Triangle* triangle, SphereInternal* sphere, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	{
		Vector3 axis;
		if (axis.set(triangle->getCenter()).sub(sphere->getCenter()).computeLengthSquared() <= (triangle->getSphereRadius() + sphere->getSphereRadius()) * (triangle->getSphereRadius() + sphere->getSphereRadius()) == false) return false;
	}

	Vector3 closestPoint;
	Vector3 axis;

	auto& sphereCenter = sphere->getCenter();
	triangle->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(sphereCenter).sub(closestPoint);
	if (checkMovementFallback(axis, movement, collision) == true) {
		return true;
	}
	auto distance = axis.computeLength() - sphere->getRadius();
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-sphere->getRadius()).add(sphere->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(closestPoint));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(BoundingBox* b1, BoundingBox* b2, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	OrientedBoundingBox obbConverted2;
	obbConverted1.fromBoundingBox(b1);
	obbConverted2.fromBoundingBox(b2);
	return doCollide(&obbConverted1, &obbConverted2, movement, collision);
}

bool CollisionDetection::doCollide(BoundingBox* aabb, Sphere* sphere, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(&obbConverted1, sphere, movement, collision);
}

bool CollisionDetection::doCollide(Sphere* sphere, BoundingBox* aabb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(sphere, &obbConverted1, movement, collision);
}

bool CollisionDetection::doCollide(BoundingBox* aabb, Capsule* capsule, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(&obbConverted1, capsule, movement, collision);
}

bool CollisionDetection::doCollide(Capsule* capsule, BoundingBox* aabb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(capsule, &obbConverted1, movement, collision);
}

bool CollisionDetection::doCollide(Sphere* s1, Sphere* s2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(s1, s2) == false) return false;

	Vector3 axis;
	Vector3 hitPoint;
	axis.set(s2->getCenter()).sub(s1->getCenter());
	if (checkMovementFallback(axis, movement, collision) == true) {
		auto collisionEntity = collision->getEntityAt(0);
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(s1->getRadius()).add(s1->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(-s2->getRadius()).add(s2->getCenter()));
		return true;
	}
	auto distance = axis.computeLength();
	auto _distance = distance - (s1->getRadius() + s2->getRadius());
	if (_distance < 0.0f) {
		auto collisionEntity = collision->addResponse(_distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(s1->getRadius()).add(s1->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-s2->getRadius()).add(s2->getCenter()));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Capsule* c1, Capsule* c2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(c1, c2) == false) return false;

	Vector3 closestPointOnCapsule1;
	Vector3 closestPointOnCapsule2;
	LineSegment::computeClosestPointsOnLineSegments(c1->getA(), c1->getB(), c2->getA(), c2->getB(), closestPointOnCapsule1, closestPointOnCapsule2);
	SphereInternal sphere1(closestPointOnCapsule1, c1->getRadius());
	SphereInternal sphere2(closestPointOnCapsule2, c2->getRadius());
	return doCollide(&sphere1, &sphere2, movement, collision);
}

bool CollisionDetection::doCollide(Capsule* c, Sphere* s, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(c, s) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;
	Vector3 hitPoint;
	auto& sphereCenter = s->getCenter();
	c->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(sphereCenter).sub(closestPoint);
	if (checkMovementFallback(axis, movement, collision) == true) {
		auto collisionEntity = collision->getEntityAt(0);
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(-s->getRadius()).add(s->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(c->getRadius()).add(closestPoint));
		return true;
	}
	auto distance = axis.computeLength();
	auto _distance = distance - (c->getRadius() + s->getRadius());
	if (_distance < 0.0f) {
		auto collisionEntity = collision->addResponse(_distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-s->getRadius()).add(s->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(c->getRadius()).add(closestPoint));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Sphere* s, Capsule* c, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(s, c) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;
	Vector3 hitPoint;
	auto& sphereCenter = s->getCenter();
	c->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(closestPoint).sub(sphereCenter);
	if (checkMovementFallback(axis, movement, collision) == true) {
		auto collisionEntity = collision->getEntityAt(0);
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(s->getRadius()).add(s->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(movement).normalize().scale(-c->getRadius()).add(closestPoint));
		return true;
	}
	auto distance = axis.computeLength();
	auto _distance = distance - (c->getRadius() + s->getRadius());
	if (_distance < 0.0f) {
		auto collisionEntity = collision->addResponse(_distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(s->getRadius()).add(s->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-c->getRadius()).add(closestPoint));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb1, OrientedBoundingBox* obb2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(obb1, obb2) == false) return false;

	// SAT best fit
	bool satHaveBestFit = false;
	Vector3 satAxisBestFit;
	float satPenetrationBestFit = 0.0f;
	Vector3 satAxis;
	float satPenetration;

	#define SAT_DETERMINE_BESTFIT() \
		if (Float::isNaN(satPenetration) == false &&  \
			(satHaveBestFit == false || -satPenetration > satPenetrationBestFit)) {  \
			satHaveBestFit = true;  \
			satAxisBestFit.set(satAxis);  \
			satPenetrationBestFit = -satPenetration;  \
		}
	//

	auto axes1 = obb1->getAxes();
	auto axes2 = obb2->getAxes();
	auto obb1Vertices = obb1->getVertices();
	auto obb2Vertices = obb2->getVertices();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes1)[0]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes1)[1]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes1)[2]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes2)[0]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes2)[1]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, satAxis.set((*axes2)[2]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[0], (*axes2)[0], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[0], (*axes2)[1], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[0], (*axes2)[2], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[1], (*axes2)[0], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[1], (*axes2)[1], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[1], (*axes2)[2], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[2], (*axes2)[0], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[2], (*axes2)[1], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(obb1Vertices, obb2Vertices, Vector3::computeCrossProduct((*axes1)[2], (*axes2)[2], satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();

	if (satHaveBestFit == false) return false;

	auto entity = collision->addResponse(satPenetrationBestFit);
	entity->getNormal().set(satAxisBestFit);
	computeHitPoints(obb1, obb2, entity);

	return true;
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, BoundingBox* aabb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(obb, &obbConverted1, movement, collision);
}

bool CollisionDetection::doCollide(BoundingBox* aabb, OrientedBoundingBox* obb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(&obbConverted1, obb, movement, collision);
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, Sphere* sphere, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(obb, sphere) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;
	auto& sphereCenter = sphere->getCenter();
	obb->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(sphereCenter).sub(closestPoint);
	float distance;
	if (axis.computeLength() < MathTools::EPSILON) {
		Vector3 pointOnFaceNearest;
		Vector3 pointOnFaceOpposite;
		obb->computeNearestPointOnFaceBoundingVolume(closestPoint, pointOnFaceNearest);
		obb->computeOppositePointOnFaceBoundingVolume(closestPoint, pointOnFaceOpposite);
		obb->computeNearestPointOnFaceBoundingVolume(pointOnFaceNearest, closestPoint);
		axis.set(sphereCenter).sub(pointOnFaceNearest);
		distance = -axis.computeLength() - sphere->getRadius();
		axis.set(pointOnFaceNearest).sub(pointOnFaceOpposite);
	} else {
		distance = axis.computeLength() - sphere->getRadius();
	}
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(closestPoint);
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-sphere->getRadius()).add(sphereCenter));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Sphere* sphere, OrientedBoundingBox* obb, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(sphere, obb) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;
	auto& sphereCenter = sphere->getCenter();
	obb->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(closestPoint).sub(sphereCenter);
	float distance;
	if (axis.computeLength() < MathTools::EPSILON) {
		Vector3 pointOnFaceNearest;
		Vector3 pointOnFaceOpposite;
		obb->computeNearestPointOnFaceBoundingVolume(closestPoint, pointOnFaceNearest);
		obb->computeOppositePointOnFaceBoundingVolume(closestPoint, pointOnFaceOpposite);
		obb->computeNearestPointOnFaceBoundingVolume(pointOnFaceNearest, closestPoint);
		axis.set(sphereCenter).sub(pointOnFaceNearest);
		distance = axis.computeLength() - sphere->getRadius();
		axis.set(pointOnFaceOpposite).sub(pointOnFaceNearest);
	} else {
		distance = axis.computeLength() - sphere->getRadius();
	}
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(sphere->getRadius()).add(sphereCenter));
		collisionEntity->addHitPoint(closestPoint);
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, Capsule* capsule, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(obb, capsule) == false) return false;

	OrientedBoundingBox obbExtended;
	Vector3 contactMin;
	Vector3 contactMax;

	auto r = capsule->getRadius();
	obbExtended.fromOrientedBoundingBox(obb);
	obbExtended.getHalfExtension().add(r);
	obbExtended.update();

	if (LineSegment::doesOrientedBoundingBoxCollideWithLineSegment(&obbExtended, capsule->getA(), capsule->getB(), contactMin, contactMax) == true) {
		Vector3 axis;
		Vector3 contactAvg;
		Vector3 closestPoint;
		Vector3 contactOptimal;
		axis.set(capsule->getB()).sub(capsule->getA()).normalize();
		contactAvg.set(contactMin).add(contactMax).scale(0.5f);
		obb->computeClosestPointOnBoundingVolume(contactAvg, closestPoint);
		if (contactAvg.equals(closestPoint) == true) {
			contactOptimal.set(contactAvg);
		} else {
			Vector3 contactAvgCollisionNormal;
			Vector3 contactAvgSubContactMin;
			contactAvgCollisionNormal.set(contactAvg).sub(closestPoint).normalize();
			auto collisionNormalDotABNormalized = Vector3::computeDotProduct(contactAvgCollisionNormal, axis);
			contactAvgSubContactMin.set(contactAvg).sub(contactMin);
			contactOptimal.set(contactAvg).sub(contactAvgSubContactMin.scale(collisionNormalDotABNormalized));
		}
		SphereInternal sphere(contactOptimal, r);
		doCollide(obb, &sphere, movement, collision);
		if (collision->hasEntitySelected() == true) return true;
	}
	return false;
}

bool CollisionDetection::doCollide(Capsule* capsule, OrientedBoundingBox* obb, const Vector3& movement, CollisionResponse* collision)
{
	if (doCollide(obb, capsule, movement, collision) == true) {
		collision->invertNormals();
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Triangle* triangle, Sphere* sphere, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle, sphere) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;

	auto& sphereCenter = sphere->getCenter();
	triangle->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(sphereCenter).sub(closestPoint);
	if (checkMovementFallback(axis, movement, collision) == true) {
		return true;
	}
	auto distance = axis.computeLength() - sphere->getRadius();
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(-sphere->getRadius()).add(sphere->getCenter()));
		collisionEntity->addHitPoint(hitPoint.set(closestPoint));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Sphere* sphere, Triangle* triangle, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(sphere, triangle) == false) return false;

	Vector3 closestPoint;
	Vector3 axis;

	auto& sphereCenter = sphere->getCenter();
	triangle->computeClosestPointOnBoundingVolume(sphereCenter, closestPoint);
	axis.set(closestPoint).sub(sphereCenter);
	if (checkMovementFallback(axis, movement, collision) == true) {
		return true;
	}
	auto distance = axis.computeLength() - sphere->getRadius();
	if (distance < 0.0f) {
		Vector3 hitPoint;
		auto collisionEntity = collision->addResponse(distance);
		collisionEntity->getNormal().set(axis).normalize();
		collisionEntity->addHitPoint(hitPoint.set(axis).normalize().scale(sphere->getRadius()).add(sphereCenter));
		collisionEntity->addHitPoint(hitPoint.set(closestPoint));
		if (CHECK_COLLISIONRESPONSE) checkCollision(collision);
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Sphere* sphere, ConvexMesh* mesh, const Vector3& movement, CollisionResponse* collision)
{
	return doCollide(sphere->getConvexMesh(), mesh, movement, collision);
}

bool CollisionDetection::doCollide(ConvexMesh* mesh, Sphere* sphere, const Vector3& movement, CollisionResponse* collision)
{
	return doCollide(mesh, sphere->getConvexMesh(), movement, collision);
}

bool CollisionDetection::doCollide(Triangle* triangle, BoundingBox* aabb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(triangle, &obbConverted1, movement, collision);
}

bool CollisionDetection::doCollide(BoundingBox* aabb, Triangle* triangle, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(&obbConverted1, triangle, movement, collision);
}

bool CollisionDetection::doCollide(Triangle* triangle, OrientedBoundingBox* obb, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle, obb) == false) return false;

	// SAT best fit
	bool satHaveBestFit = false;
	Vector3 satAxisBestFit;
	float satPenetrationBestFit = 0.0f;
	Vector3 satAxis;
	float satPenetration;

	#define SAT_DETERMINE_BESTFIT() \
		if (Float::isNaN(satPenetration) == false &&  \
			(satHaveBestFit == false || -satPenetration > satPenetrationBestFit)) {  \
			satHaveBestFit = true;  \
			satAxisBestFit.set(satAxis);  \
			satPenetrationBestFit = -satPenetration;  \
		}
	//

	Vector3 triangle1Edge1;
	Vector3 triangle1Edge2;
	Vector3 triangle1Edge3;
	Vector3 triangle1Normal;
	auto obbVertices = obb->getVertices();
	auto obbAxes = obb->getAxes();
	auto triangleVertices = triangle->getVertices();
	triangle1Edge1.set((*triangleVertices)[1]).sub((*triangleVertices)[0]).normalize();
	triangle1Edge2.set((*triangleVertices)[2]).sub((*triangleVertices)[1]).normalize();
	triangle1Edge3.set((*triangleVertices)[0]).sub((*triangleVertices)[2]).normalize();
	Vector3::computeCrossProduct(triangle1Edge1, triangle1Edge2, triangle1Normal).normalize();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, satAxis.set(triangle1Normal), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, satAxis.set((*obbAxes)[0]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, satAxis.set((*obbAxes)[1]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, satAxis.set((*obbAxes)[2]), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();
	if (SeparatingAxisTheorem::doSpanIntersect(triangleVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT();

	if (satHaveBestFit == false) return false;

	auto entity = collision->addResponse(satPenetrationBestFit);
	entity->getNormal().set(satAxisBestFit);
	computeHitPoints(triangle, obb, entity);

	return true;
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, Triangle* triangle, const Vector3& movement, CollisionResponse* collision)
{
	if (doCollide(triangle, obb, movement, collision) == true) {
		collision->invertNormals();
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(ConvexMesh* mesh, BoundingBox* aabb, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(mesh, &obbConverted1, movement, collision);
}

bool CollisionDetection::doCollide(BoundingBox* aabb, ConvexMesh* mesh, const Vector3& movement, CollisionResponse* collision)
{
	OrientedBoundingBox obbConverted1;
	obbConverted1.fromBoundingBox(aabb);
	return doCollide(&obbConverted1, mesh, movement, collision);
}

bool CollisionDetection::doCollide(ConvexMesh* mesh, OrientedBoundingBox* obb, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(mesh, obb) == false) return false;

	// SAT best fit
	bool satHaveBestFit = false;
	Vector3 satAxisBestFit;
	float satPenetrationBestFit = 0.0f;
	Vector3 satAxis;
	float satPenetration;

	#define SAT_DETERMINE_BESTFIT() \
		if (Float::isNaN(satPenetration) == false &&  \
			(satHaveBestFit == false || -satPenetration > satPenetrationBestFit)) {  \
			satHaveBestFit = true;  \
			satAxisBestFit.set(satAxis);  \
			satPenetrationBestFit = -satPenetration;  \
		}
	//

	Vector3 triangle1Edge1;
	Vector3 triangle1Edge2;
	Vector3 triangle1Edge3;
	Vector3 triangle1Normal;
	vector<Triangle*> testTriangles;

	auto obbVertices = obb->getVertices();
	auto obbAxes = obb->getAxes();
	auto meshVertices = mesh->getVertices();

	for (auto& triangle : *mesh->getTriangles()) {

		auto triangleVertices = triangle.getVertices();

		triangle1Edge1.set((*triangleVertices)[1]).sub((*triangleVertices)[0]).normalize();
		triangle1Edge2.set((*triangleVertices)[2]).sub((*triangleVertices)[1]).normalize();
		triangle1Edge3.set((*triangleVertices)[0]).sub((*triangleVertices)[2]).normalize();
		Vector3::computeCrossProduct(triangle1Edge1, triangle1Edge2, triangle1Normal).normalize();

		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set(triangle1Normal), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set((*obbAxes)[0]), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set((*obbAxes)[1]), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set((*obbAxes)[2]), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set(triangle1Edge1), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set(triangle1Edge2), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, satAxis.set(triangle1Edge3), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[0], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[1], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge1, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge2, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		if (SeparatingAxisTheorem::doSpanIntersect(meshVertices, obbVertices, Vector3::computeCrossProduct((*obbAxes)[2], triangle1Edge3, satAxis).normalize(), satPenetration) == false) return false;
		SAT_DETERMINE_BESTFIT();
		testTriangles.push_back(&triangle);
	}

	if (satHaveBestFit == false) return false;

	auto entity = collision->addResponse(satPenetrationBestFit);
	entity->getNormal().set(satAxisBestFit);
	for (auto i = 0; i < testTriangles.size(); i++) {
		computeHitPoints(testTriangles[i], obb, entity);
	}

	return true;
}

bool CollisionDetection::doCollide(OrientedBoundingBox* obb, ConvexMesh* mesh, const Vector3& movement, CollisionResponse* collision)
{
	if (doCollide(mesh, obb, movement, collision) == true) {
		collision->invertNormals();
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Triangle* triangle1, Triangle* triangle2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle1, triangle2) == false) return false;

	Vector3 triangle1Edge1;
	Vector3 triangle1Edge2;
	Vector3 triangle1Edge3;
	Vector3 triangle1Normal;

	auto triangle1Vertices = triangle1->getVertices();
	triangle1Edge1.set((*triangle1Vertices)[1]).sub((*triangle1Vertices)[0]).normalize();
	triangle1Edge2.set((*triangle1Vertices)[2]).sub((*triangle1Vertices)[1]).normalize();
	triangle1Edge3.set((*triangle1Vertices)[0]).sub((*triangle1Vertices)[2]).normalize();
	Vector3::computeCrossProduct(triangle1Edge1, triangle1Edge2, triangle1Normal).normalize();

	Vector3 triangle2Edge1;
	Vector3 triangle2Edge2;
	Vector3 triangle2Edge3;
	Vector3 triangle2Normal;

	auto triangle2Vertices = triangle2->getVertices();
	triangle2Edge1.set((*triangle2Vertices)[1]).sub((*triangle2Vertices)[0]).normalize();
	triangle2Edge2.set((*triangle2Vertices)[2]).sub((*triangle2Vertices)[1]).normalize();
	triangle2Edge3.set((*triangle2Vertices)[0]).sub((*triangle2Vertices)[2]).normalize();
	Vector3::computeCrossProduct(triangle2Edge1, triangle2Edge2, triangle2Normal).normalize();

	// SAT best fit
	bool satHaveBestFit = false;
	Vector3 satAxisBestFit;
	float satPenetrationBestFit = 0.0f;
	Vector3 satAxis;
	float satPenetration;

	#define SAT_DETERMINE_BESTFIT() \
		if (Float::isNaN(satPenetration) == false &&  \
			(satHaveBestFit == false || -satPenetration > satPenetrationBestFit)) {  \
			satHaveBestFit = true;  \
			satAxisBestFit.set(satAxis);  \
			satPenetrationBestFit = -satPenetration;  \
		}
	//

	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, satAxis.set(triangle1Normal), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, satAxis.set(triangle2Normal), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()
	if (SeparatingAxisTheorem::doSpanIntersect(triangle1Vertices, triangle2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
	SAT_DETERMINE_BESTFIT()

	if (satHaveBestFit == false) return false;

	auto entity = collision->addResponse(satPenetrationBestFit);
	entity->getNormal().set(satAxisBestFit);
	computeHitPoints(triangle1, triangle2, entity);

	return true;
}

bool CollisionDetection::doCollide(ConvexMesh* mesh1, ConvexMesh* mesh2, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(mesh1, mesh2) == false) return false;

	Vector3 triangle1Edge1;
	Vector3 triangle1Edge2;
	Vector3 triangle1Edge3;
	Vector3 triangle1Normal;
	Vector3 triangle2Edge1;
	Vector3 triangle2Edge2;
	Vector3 triangle2Edge3;
	Vector3 triangle2Normal;

	// SAT best fit
	bool satHaveBestFit = false;
	bool satHaveBestFitTerrain = false;
	Vector3 satAxisBestFit;
	Vector3 satAxisBestFitTerrain;
	float satPenetrationBestFit = 0.0f;
	float satPenetrationBestFitTerrain = 0.0f;
	Vector3 satAxis;
	float satPenetration;
	bool terrain = mesh1->isTerrain() == true || mesh2->isTerrain() == true;

	#define SAT_DETERMINE_BESTFIT() \
		if (Float::isNaN(satPenetration) == false) { \
			if (terrain == true && Math::abs(satAxis.getY()) > 0.2f && \
				(satHaveBestFitTerrain == false || -satPenetration > satPenetrationBestFitTerrain)) { \
				satHaveBestFitTerrain = true; \
				satAxisBestFitTerrain.set(satAxis); \
				satPenetrationBestFitTerrain = -satPenetration; \
			} else \
			if (satHaveBestFit == false || -satPenetration > satPenetrationBestFit) { \
				satHaveBestFit = true; \
				satAxisBestFit.set(satAxis); \
				satPenetrationBestFit = -satPenetration; \
			} \
		}

	vector<Triangle*> testTriangles;

	auto mesh1Vertices = mesh1->getVertices();
	auto mesh2Vertices = mesh2->getVertices();

	for (auto& triangle1 : *mesh1->getTriangles()) {
		for (auto& triangle2 : *mesh2->getTriangles()) {
			auto triangle1Vertices = triangle1.getVertices();
			triangle1Edge1.set((*triangle1Vertices)[1]).sub((*triangle1Vertices)[0]).normalize();
			triangle1Edge2.set((*triangle1Vertices)[2]).sub((*triangle1Vertices)[1]).normalize();
			triangle1Edge3.set((*triangle1Vertices)[0]).sub((*triangle1Vertices)[2]).normalize();
			Vector3::computeCrossProduct(triangle1Edge1, triangle1Edge2, triangle1Normal).normalize();
			auto triangle2Vertices = triangle2.getVertices();
			triangle2Edge1.set((*triangle2Vertices)[1]).sub((*triangle2Vertices)[0]).normalize();
			triangle2Edge2.set((*triangle2Vertices)[2]).sub((*triangle2Vertices)[1]).normalize();
			triangle2Edge3.set((*triangle2Vertices)[0]).sub((*triangle2Vertices)[2]).normalize();
			Vector3::computeCrossProduct(triangle2Edge1, triangle2Edge2, triangle2Normal).normalize();

			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, satAxis.set(triangle1Normal), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, satAxis.set(triangle2Normal), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge1, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge2, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge1, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge2, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();
			if (SeparatingAxisTheorem::doSpanIntersect(mesh1Vertices, mesh2Vertices, Vector3::computeCrossProduct(triangle1Edge3, triangle2Edge3, satAxis).normalize(), satPenetration) == false) return false;
			SAT_DETERMINE_BESTFIT();

			testTriangles.push_back(&triangle1);
			testTriangles.push_back(&triangle2);
		}
	}

	if (terrain == true && satHaveBestFitTerrain == true) {
		auto entity = collision->addResponse(satPenetrationBestFitTerrain);
		entity->getNormal().set(satAxisBestFitTerrain);
		for (auto i = 0; i < testTriangles.size() / 2; i++) {
			computeHitPoints(testTriangles[i * 2 + 0], testTriangles[i * 2 + 1], entity);
		}
		return true;
	}
	if (satHaveBestFit == true) {
		auto entity = collision->addResponse(satPenetrationBestFit);
		entity->getNormal().set(satAxisBestFit);
		for (auto i = 0; i < testTriangles.size() / 2; i++) {
			computeHitPoints(testTriangles[i * 2 + 0], testTriangles[i * 2 + 1], entity);
		}
		return true;
	}

	return false;
}

bool CollisionDetection::doCollide(Triangle* triangle, ConvexMesh* mesh, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle, mesh) == false) return false;

	CollisionResponse collision1;
	for (auto& triangle2 : *mesh->getTriangles()) {
		if (doCollide(triangle, &triangle2, movement, &collision1) == true) {
			if (collision->hasEntitySelected() == false || collision1.getPenetration() > collision->getPenetration()) {
				collision->fromResponse(&collision1);
			}
		}
	}
	return collision->hasEntitySelected();
}

bool CollisionDetection::doCollide(ConvexMesh* mesh, Triangle* triangle, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle, mesh) == false) return false;

	CollisionResponse collision1;
	for (auto& triangle2 : *mesh->getTriangles()) {
		if (doCollide(&triangle2, triangle, movement, &collision1) == true) {
			if (collision->hasEntitySelected() == false || collision1.getPenetration() > collision->getPenetration()) {
				collision->fromResponse(&collision1);
			}
		}
	}
	return collision->hasEntitySelected();
}

bool CollisionDetection::doCollide(Triangle* triangle, Capsule* capsule, const Vector3& movement, CollisionResponse* collision)
{
	collision->reset();
	if (doBroadTest(triangle, capsule) == false) return false;

	array<Vector3, 5> closestPointsOnCapsuleSegment;
	array<Vector3, 5> closestPointsOnTriangleSegments;
	Vector3 closestPoint;

	auto triangleVertices = triangle->getVertices();
	LineSegment::computeClosestPointsOnLineSegments(capsule->getA(), capsule->getB(), (*triangleVertices)[1], (*triangleVertices)[0], closestPointsOnCapsuleSegment[0], closestPointsOnTriangleSegments[0]);
	LineSegment::computeClosestPointsOnLineSegments(capsule->getA(), capsule->getB(), (*triangleVertices)[2], (*triangleVertices)[1], closestPointsOnCapsuleSegment[1], closestPointsOnTriangleSegments[1]);
	LineSegment::computeClosestPointsOnLineSegments(capsule->getA(), capsule->getB(), (*triangleVertices)[0], (*triangleVertices)[2], closestPointsOnCapsuleSegment[2], closestPointsOnTriangleSegments[2]);
	closestPointsOnCapsuleSegment[3].set(capsule->getA());
	triangle->computeClosestPointOnBoundingVolume(closestPointsOnCapsuleSegment[3], closestPointsOnTriangleSegments[3]);
	closestPointsOnCapsuleSegment[4].set(capsule->getB());
	triangle->computeClosestPointOnBoundingVolume(closestPointsOnCapsuleSegment[4], closestPointsOnTriangleSegments[4]);
	auto bestFitLength = closestPoint.set(closestPointsOnCapsuleSegment[0]).sub(closestPointsOnTriangleSegments[0]).computeLengthSquared();
	auto bestFitTest = 0;
	for (auto i = 1; i < 5; i++) {
		auto testLength = closestPoint.set(closestPointsOnCapsuleSegment[i]).sub(closestPointsOnTriangleSegments[i]).computeLengthSquared();
		if (testLength < bestFitLength) {
			bestFitLength = testLength;
			bestFitTest = i;
		}
	}
	SphereInternal sphere1(closestPointsOnCapsuleSegment[bestFitTest], capsule->getRadius());
	return doCollide(triangle, &sphere1, movement, collision);
}

bool CollisionDetection::doCollide(Capsule* capsule, Triangle* triangle, const Vector3& movement, CollisionResponse* collision)
{
	if (doCollide(triangle, capsule, movement, collision) == true) {
		collision->invertNormals();
		return true;
	} else {
		return false;
	}
}

bool CollisionDetection::doCollide(Capsule* capsule, ConvexMesh* mesh, const Vector3& movement, CollisionResponse* collision)
{
	return doCollide(capsule->getConvexMesh(), mesh, movement, collision);
}

bool CollisionDetection::doCollide(ConvexMesh* mesh, Capsule* capsule, const Vector3& movement, CollisionResponse* collision)
{
	return doCollide(mesh, capsule->getConvexMesh(), movement, collision);
}

void CollisionDetection::computeHitPoints(OrientedBoundingBox* obb1, OrientedBoundingBox* obb2, CollisionResponse_Entity* collisionEntity)
{
	Vector3 hitPointTriangle1;
	Vector3 hitPointTriangle2;
	auto obb1Vertices = obb1->getVertices();
	auto obb1FacesVerticesIndexes = OrientedBoundingBox::getFacesVerticesIndexes();
	auto obb2Vertices = obb2->getVertices();
	auto obb2FacesVerticesIndexes = OrientedBoundingBox::getFacesVerticesIndexes();
	for (auto triangleObb1Idx = 0; triangleObb1Idx < obb1FacesVerticesIndexes->size(); triangleObb1Idx++)
		for (auto triangleObb2Idx = 0; triangleObb2Idx < obb2FacesVerticesIndexes->size(); triangleObb2Idx++) {
			auto tritriReturn = TriangleTriangleIntersection::computeTriangleTriangleIntersection(
				(*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][0]],
				(*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][1]],
				(*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][2]],
				(*obb2Vertices)[(*obb2FacesVerticesIndexes)[triangleObb2Idx][0]],
				(*obb2Vertices)[(*obb2FacesVerticesIndexes)[triangleObb2Idx][1]],
				(*obb2Vertices)[(*obb2FacesVerticesIndexes)[triangleObb2Idx][2]],
				hitPointTriangle1,
				hitPointTriangle2
			);
			{
				{
					auto v = tritriReturn;
					if ((v == TriangleTriangleIntersection::NOINTERSECTION)) {
						goto end_switch0;;
					}
					if ((v == TriangleTriangleIntersection::INTERSECTION)) {
						collisionEntity->addHitPoint(hitPointTriangle1);
						collisionEntity->addHitPoint(hitPointTriangle2);
						goto end_switch0;;
					}
					if ((v == TriangleTriangleIntersection::COPLANAR_INTERSECTION)) {
						Triangle triangle1;
						Triangle triangle2;
						auto _triangle1Vertices = triangle1.getVertices();
						auto _triangle2Vertices = triangle2.getVertices();
						(*_triangle1Vertices)[0].set((*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][0]]);
						(*_triangle1Vertices)[1].set((*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][1]]);
						(*_triangle1Vertices)[2].set((*obb1Vertices)[(*obb1FacesVerticesIndexes)[triangleObb1Idx][2]]);
						triangle1.update();
						(*_triangle2Vertices)[0].set((*obb2Vertices)[(*obb1FacesVerticesIndexes)[triangleObb2Idx][0]]);
						(*_triangle2Vertices)[1].set((*obb2Vertices)[(*obb1FacesVerticesIndexes)[triangleObb2Idx][1]]);
						(*_triangle2Vertices)[2].set((*obb2Vertices)[(*obb1FacesVerticesIndexes)[triangleObb2Idx][2]]);
						triangle2.update();
						computeCoplanarTrianglesHitPoints(&triangle1, &triangle2, collisionEntity);
						goto end_switch0;;
					}
					end_switch0:;
				}
			}

		}

}

void CollisionDetection::computeHitPoints(Triangle* triangle, OrientedBoundingBox* obb, CollisionResponse_Entity* collisionEntity)
{
	Vector3 hitPointTriangle1;
	Vector3 hitPointTriangle2;
	auto triangleVertices = triangle->getVertices();
	auto obbVertices = obb->getVertices();
	auto obbFacesVerticesIndexes = OrientedBoundingBox::getFacesVerticesIndexes();
	for (auto triangleObbIdx = 0; triangleObbIdx < obbFacesVerticesIndexes->size(); triangleObbIdx++) {
		auto tritriReturn = TriangleTriangleIntersection::computeTriangleTriangleIntersection(
			(*triangleVertices)[0],
			(*triangleVertices)[1],
			(*triangleVertices)[2],
			(*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][0]],
			(*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][1]],
			(*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][2]],
			hitPointTriangle1,
			hitPointTriangle2
		);
		{
			{
				auto v = tritriReturn;
				if ((v == TriangleTriangleIntersection::NOINTERSECTION)) {
					goto end_switch1;;
				}
				if ((v == TriangleTriangleIntersection::INTERSECTION)) {
					collisionEntity->addHitPoint(hitPointTriangle1);
					collisionEntity->addHitPoint(hitPointTriangle2);
					goto end_switch1;;
				}
				if ((v == TriangleTriangleIntersection::COPLANAR_INTERSECTION)) {
					Triangle triangle1;
					Triangle triangle2;
					auto _triangle1Vertices = triangle1.getVertices();
					auto _triangle2Vertices = triangle2.getVertices();
					(*_triangle1Vertices)[0].set((*triangleVertices)[0]);
					(*_triangle1Vertices)[1].set((*triangleVertices)[1]);
					(*_triangle1Vertices)[2].set((*triangleVertices)[2]);
					triangle1.update();
					(*_triangle2Vertices)[0].set((*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][0]]);
					(*_triangle2Vertices)[1].set((*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][1]]);
					(*_triangle2Vertices)[2].set((*obbVertices)[(*obbFacesVerticesIndexes)[triangleObbIdx][2]]);
					triangle2.update();
					computeCoplanarTrianglesHitPoints(&triangle1, &triangle2, collisionEntity);
					goto end_switch1;;
				}
				end_switch1:;
			}
		}

	}
}

void CollisionDetection::computeHitPoints(Triangle* triangle1, Triangle* triangle2, CollisionResponse_Entity* collisionEntity)
{
	Vector3 hitPointTriangle1;
	Vector3 hitPointTriangle2;
	auto triangle1Vertices = triangle1->getVertices();
	auto triangle2Vertices = triangle2->getVertices();
	auto tritriReturn = TriangleTriangleIntersection::computeTriangleTriangleIntersection(
		(*triangle1Vertices)[0],
		(*triangle1Vertices)[1],
		(*triangle1Vertices)[2],
		(*triangle2Vertices)[0],
		(*triangle2Vertices)[1],
		(*triangle2Vertices)[2],
		hitPointTriangle1,
		hitPointTriangle2
	);
	{
		{
			auto v = tritriReturn;
			if ((v == TriangleTriangleIntersection::NOINTERSECTION)) {
				goto end_switch2;;
			}
			if ((v == TriangleTriangleIntersection::INTERSECTION)) {
				collisionEntity->addHitPoint(hitPointTriangle1);
				collisionEntity->addHitPoint(hitPointTriangle2);
				goto end_switch2;;
			}
			if ((v == TriangleTriangleIntersection::COPLANAR_INTERSECTION)) {
				auto _triangle1Vertices = triangle1->getVertices();
				auto _triangle2Vertices = triangle2->getVertices();
				(*_triangle1Vertices)[0].set((*triangle1Vertices)[0]);
				(*_triangle1Vertices)[1].set((*triangle1Vertices)[1]);
				(*_triangle1Vertices)[2].set((*triangle1Vertices)[2]);
				triangle1->update();
				(*_triangle2Vertices)[0].set((*triangle2Vertices)[0]);
				(*_triangle2Vertices)[1].set((*triangle2Vertices)[1]);
				(*_triangle2Vertices)[2].set((*triangle2Vertices)[2]);
				triangle2->update();
				computeCoplanarTrianglesHitPoints(triangle1, triangle2, collisionEntity);
				goto end_switch2;;
			}
			end_switch2:;
		}
	}

}

void CollisionDetection::computeCoplanarTrianglesHitPoints(Triangle* triangle1, Triangle* triangle2, CollisionResponse_Entity* collisionEntity)
{
	Vector3 hitPoint;
	auto triangle1Vertices = triangle1->getVertices();
	auto triangle2Vertices = triangle2->getVertices();
	auto lineSegmentsIntersections = 0;
	for (auto i = 0; i < LINESEGMENTSTRIANGLEINDICES.size(); i += 2)
		for (auto j = 0; j < LINESEGMENTSTRIANGLEINDICES.size(); j += 2) {
			if (LineSegment::doesLineSegmentsCollide(
					(*triangle1Vertices)[LINESEGMENTSTRIANGLEINDICES[i + 0]],
					(*triangle1Vertices)[LINESEGMENTSTRIANGLEINDICES[i + 1]],
					(*triangle2Vertices)[LINESEGMENTSTRIANGLEINDICES[j + 0]],
					(*triangle2Vertices)[LINESEGMENTSTRIANGLEINDICES[j + 1]],
					hitPoint
				) == true) {
				lineSegmentsIntersections++;
				collisionEntity->addHitPoint(hitPoint);
			}
		}

	if (lineSegmentsIntersections == 0) {
		for (auto i = 0; i < triangle1Vertices->size(); i++) {
			if (triangle2->containsPoint((*triangle1Vertices)[i]) == true) {
				collisionEntity->addHitPoint((*triangle1Vertices)[i]);
			}
		}
		for (auto i = 0; i < triangle2Vertices->size(); i++) {
			if (triangle1->containsPoint((*triangle2Vertices)[i]) == true) {
				collisionEntity->addHitPoint((*triangle2Vertices)[i]);
			}
		}
	}
}

void CollisionDetection::checkCollision(CollisionResponse* collision)
{
	auto& normalXYZ = collision->getNormal()->getArray();
	if (Float::isNaN(normalXYZ[0]) == true || Float::isNaN(normalXYZ[1]) == true || Float::isNaN(normalXYZ[2]) == true) {
		Console::println(string("CollisionDetection::checkCollision(): BROKEN NORMAL"));
	}
}
