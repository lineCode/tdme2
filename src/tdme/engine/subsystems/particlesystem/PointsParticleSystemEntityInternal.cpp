// Generated from /tdme/src/tdme/engine/subsystems/particlesystem/PointsParticleSystemEntityInternal.java
#include <tdme/engine/subsystems/particlesystem/PointsParticleSystemEntityInternal.h>

#include <java/lang/ArrayStoreException.h>
#include <java/lang/ClassCastException.h>
#include <java/lang/Math.h>
#include <java/lang/NullPointerException.h>
#include <java/lang/Object.h>
#include <java/lang/String.h>
#include <java/lang/System.h>
#include <java/util/Iterator.h>
#include <tdme/engine/Engine.h>
#include <tdme/engine/Entity.h>
#include <tdme/engine/Partition.h>
#include <tdme/engine/Timing.h>
#include <tdme/engine/Transformations.h>
#include <tdme/engine/model/Color4.h>
#include <tdme/engine/primitives/BoundingBox.h>
#include <tdme/engine/subsystems/object/TransparentRenderPointsPool.h>
#include <tdme/engine/subsystems/particlesystem/Particle.h>
#include <tdme/engine/subsystems/particlesystem/ParticleEmitter.h>
#include <tdme/engine/subsystems/particlesystem/ParticleSystemEntity.h>
#include <tdme/engine/subsystems/renderer/GLRenderer.h>
#include <tdme/math/MathTools.h>
#include <tdme/math/Matrix4x4.h>
#include <tdme/math/Vector3.h>
#include <tdme/utils/ArrayListIteratorMultiple.h>
#include <Array.h>
#include <ObjectArray.h>
#include <SubArray.h>

using tdme::engine::subsystems::particlesystem::PointsParticleSystemEntityInternal;
using java::lang::ArrayStoreException;
using java::lang::ClassCastException;
using java::lang::Math;
using java::lang::NullPointerException;
using java::lang::Object;
using java::lang::String;
using java::lang::System;
using java::util::Iterator;
using tdme::engine::Engine;
using tdme::engine::Entity;
using tdme::engine::Partition;
using tdme::engine::Timing;
using tdme::engine::Transformations;
using tdme::engine::model::Color4;
using tdme::engine::primitives::BoundingBox;
using tdme::engine::subsystems::object::TransparentRenderPointsPool;
using tdme::engine::subsystems::particlesystem::Particle;
using tdme::engine::subsystems::particlesystem::ParticleEmitter;
using tdme::engine::subsystems::particlesystem::ParticleSystemEntity;
using tdme::engine::subsystems::renderer::GLRenderer;
using tdme::math::MathTools;
using tdme::math::Matrix4x4;
using tdme::math::Vector3;
using tdme::utils::ArrayListIteratorMultiple;

template<typename ComponentType, typename... Bases> struct SubArray;
namespace tdme {
namespace engine {
namespace subsystems {
namespace particlesystem {
typedef ::SubArray< ::tdme::engine::subsystems::particlesystem::Particle, ::java::lang::ObjectArray > ParticleArray;
}  // namespace particlesystem
}  // namespace subsystems
}  // namespace engine
}  // namespace tdme

template<typename T, typename U>
static T java_cast(U* u)
{
    if (!u) return static_cast<T>(nullptr);
    auto t = dynamic_cast<T>(u);
    if (!t) throw new ::java::lang::ClassCastException();
    return t;
}

PointsParticleSystemEntityInternal::PointsParticleSystemEntityInternal(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
	clinit();
}

PointsParticleSystemEntityInternal::PointsParticleSystemEntityInternal(String* id, bool doCollisionTests, ParticleEmitter* emitter, int32_t maxPoints, bool autoEmit) 
	: PointsParticleSystemEntityInternal(*static_cast< ::default_init_tag* >(0))
{
	ctor(id,doCollisionTests,emitter,maxPoints,autoEmit);
}

void PointsParticleSystemEntityInternal::ctor(String* id, bool doCollisionTests, ParticleEmitter* emitter, int32_t maxPoints, bool autoEmit)
{
	super::ctor();
	this->id = id;
	this->enabled = true;
	this->doCollisionTests = doCollisionTests;
	this->active = false;
	this->emitter = emitter;
	particles = new ParticleArray(maxPoints);
	for (auto i = 0; i < particles->length; i++) {
		particles->set(i, new Particle());
	}
	this->maxPoints = maxPoints;
	velocityForTime = new Vector3();
	point = new Vector3();
	boundingBox = new BoundingBox();
	boundingBoxTransformed = new BoundingBox();
	inverseTransformation = new Transformations();
	this->effectColorMul = new Color4(1.0f, 1.0f, 1.0f, 1.0f);
	this->effectColorAdd = new Color4(0.0f, 0.0f, 0.0f, 0.0f);
	this->pickable = false;
	this->autoEmit = autoEmit;
	this->particlesToSpawnRemainder = 0.0f;
}

String* PointsParticleSystemEntityInternal::getId()
{
	return id;
}

void PointsParticleSystemEntityInternal::setRenderer(GLRenderer* renderer)
{
	this->renderer = renderer;
	this->pointsRenderPool = new TransparentRenderPointsPool(maxPoints);
}

void PointsParticleSystemEntityInternal::setEngine(Engine* engine)
{
	this->engine = engine;
}

bool PointsParticleSystemEntityInternal::isEnabled()
{
	return enabled;
}

bool PointsParticleSystemEntityInternal::isActive()
{
	return active;
}

void PointsParticleSystemEntityInternal::setEnabled(bool enabled)
{
	this->enabled = enabled;
}

Color4* PointsParticleSystemEntityInternal::getEffectColorMul()
{
	return effectColorMul;
}

Color4* PointsParticleSystemEntityInternal::getEffectColorAdd()
{
	return effectColorAdd;
}

bool PointsParticleSystemEntityInternal::isPickable()
{
	return pickable;
}

void PointsParticleSystemEntityInternal::setPickable(bool pickable)
{
	this->pickable = pickable;
}

bool PointsParticleSystemEntityInternal::isAutoEmit()
{
	return autoEmit;
}

void PointsParticleSystemEntityInternal::setAutoEmit(bool autoEmit)
{
	this->autoEmit = autoEmit;
}

bool PointsParticleSystemEntityInternal::isDynamicShadowingEnabled()
{
	return false;
}

void PointsParticleSystemEntityInternal::setDynamicShadowingEnabled(bool dynamicShadowing)
{
}

void PointsParticleSystemEntityInternal::update()
{
	super::update();
	emitter->fromTransformations(this);
	inverseTransformation->getTransformationsMatrix()->set(this->getTransformationsMatrix())->invert();
}

void PointsParticleSystemEntityInternal::fromTransformations(Transformations* transformations)
{
	super::fromTransformations(transformations);
	emitter->fromTransformations(transformations);
	inverseTransformation->getTransformationsMatrix()->set(this->getTransformationsMatrix())->invert();
}

void PointsParticleSystemEntityInternal::updateParticles()
{
	if (enabled == false || active == false)
		return;

	auto bbMinXYZ = boundingBoxTransformed->getMin()->getArray();
	auto bbMaxXYZ = boundingBoxTransformed->getMax()->getArray();
	auto haveBoundingBox = false;
	float distanceFromCamera;
	auto modelViewMatrix = renderer->getModelViewMatrix();
	distanceFromCamera = -point->getZ();
	pointsRenderPool->reset();
	auto activeParticles = 0;
	auto timeDelta = engine->getTiming()->getDeltaTime();
	for (auto i = 0; i < particles->length; i++) {
		auto particle = (*particles)[i];
		if (particle->active == false)
			continue;

		particle->lifeTimeCurrent += timeDelta;
		if (particle->lifeTimeCurrent >= particle->lifeTimeMax) {
			particle->active = false;
			continue;
		}
		if (particle->mass > MathTools::EPSILON)
			particle->velocity->subY(0.5f * MathTools::g * static_cast< float >(timeDelta) / 1000.0f);

		particle->position->add(velocityForTime->set(particle->velocity)->scale(static_cast< float >(timeDelta) / 1000.0f));
		auto color = particle->color->getArray();
		auto colorAdd = particle->colorAdd->getArray();
		(*color)[0] += (*colorAdd)[0] * static_cast< float >(timeDelta);
		(*color)[1] += (*colorAdd)[1] * static_cast< float >(timeDelta);
		(*color)[2] += (*colorAdd)[2] * static_cast< float >(timeDelta);
		(*color)[3] += (*colorAdd)[3] * static_cast< float >(timeDelta);
		modelViewMatrix->multiply(particle->position, point);
		if (doCollisionTests == true) {
			for (auto _i = engine->getPartition()->getObjectsNearTo(particle->position)->iterator(); _i->hasNext(); ) {
				Entity* entity = java_cast< Entity* >(_i->next());
				{
					if (static_cast< Object* >(entity) == static_cast< Object* >(this))
						continue;

					if (dynamic_cast< ParticleSystemEntity* >(entity) != nullptr)
						continue;

					if (entity->getBoundingBoxTransformed()->containsPoint(particle->position)) {
						particle->active = false;
						continue;
					}
				}
			}
		}
		activeParticles++;
		distanceFromCamera = -point->getZ();
		auto positionXYZ = particle->position->getArray();
		if (haveBoundingBox == false) {
			System::arraycopy(positionXYZ, 0, bbMinXYZ, 0, 3);
			System::arraycopy(positionXYZ, 0, bbMaxXYZ, 0, 3);
			haveBoundingBox = true;
		} else {
			if ((*positionXYZ)[0] < (*bbMinXYZ)[0])
				(*bbMinXYZ)[0] = (*positionXYZ)[0];

			if ((*positionXYZ)[1] < (*bbMinXYZ)[1])
				(*bbMinXYZ)[1] = (*positionXYZ)[1];

			if ((*positionXYZ)[2] < (*bbMinXYZ)[2])
				(*bbMinXYZ)[2] = (*positionXYZ)[2];

			if ((*positionXYZ)[0] > (*bbMaxXYZ)[0])
				(*bbMaxXYZ)[0] = (*positionXYZ)[0];

			if ((*positionXYZ)[1] > (*bbMaxXYZ)[1])
				(*bbMaxXYZ)[1] = (*positionXYZ)[1];

			if ((*positionXYZ)[2] > (*bbMaxXYZ)[2])
				(*bbMaxXYZ)[2] = (*positionXYZ)[2];

		}
		pointsRenderPool->addPoint(point, particle->color, distanceFromCamera);
	}
	if (activeParticles == 0) {
		active = false;
		return;
	}
	boundingBoxTransformed->getMin()->sub(0.05f);
	boundingBoxTransformed->getMax()->add(0.05f);
	boundingBoxTransformed->update();
	boundingBox->fromBoundingVolumeWithTransformations(boundingBoxTransformed, inverseTransformation);
}

void PointsParticleSystemEntityInternal::dispose()
{
}

ParticleEmitter* PointsParticleSystemEntityInternal::getParticleEmitter()
{
	return emitter;
}

int32_t PointsParticleSystemEntityInternal::emitParticles()
{
	active = true;
	auto timeDelta = engine->getTiming()->getDeltaTime();
	auto particlesToSpawnInteger = 0;
	if (autoEmit == true) {
		auto particlesToSpawn = emitter->getCount() * engine->getTiming()->getDeltaTime() / 1000.0f;
		particlesToSpawnInteger = static_cast< int32_t >(particlesToSpawn);
		particlesToSpawnRemainder += particlesToSpawn - particlesToSpawnInteger;
		if (particlesToSpawnRemainder > 1.0f) {
			particlesToSpawn += 1.0f;
			particlesToSpawnInteger++;
			particlesToSpawnRemainder -= 1.0f;
		}
	} else {
		particlesToSpawnInteger = emitter->getCount();
	}
	if (particlesToSpawnInteger == 0)
		return 0;

	auto particlesSpawned = 0;
	for (auto i = 0; i < particles->length; i++) {
		auto particle = (*particles)[i];
		if (particle->active == true)
			continue;

		emitter->emit(particle);
		auto timeDeltaRnd = static_cast< int64_t >((Math::random() * static_cast< double >(timeDelta)));
		if (particle->mass > MathTools::EPSILON)
			particle->velocity->subY(0.5f * MathTools::g * static_cast< float >(timeDeltaRnd) / 1000.0f);

		particle->position->add(velocityForTime->set(particle->velocity)->scale(static_cast< float >(timeDeltaRnd) / 1000.0f));
		particlesSpawned++;
		if (particlesSpawned == particlesToSpawnInteger)
			break;

	}
	return particlesSpawned;
}

TransparentRenderPointsPool* PointsParticleSystemEntityInternal::getRenderPointsPool()
{
	return pointsRenderPool;
}

extern java::lang::Class* class_(const char16_t* c, int n);

java::lang::Class* PointsParticleSystemEntityInternal::class_()
{
    static ::java::lang::Class* c = ::class_(u"tdme.engine.subsystems.particlesystem.PointsParticleSystemEntityInternal", 72);
    return c;
}

java::lang::Class* PointsParticleSystemEntityInternal::getClass0()
{
	return class_();
}

