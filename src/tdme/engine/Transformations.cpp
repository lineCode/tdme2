// Generated from /tdme/src/tdme/engine/Transformations.java
#include <tdme/engine/Transformations.h>

#include <java/lang/Object.h>
#include <java/lang/String.h>
#include <java/lang/StringBuilder.h>
#include <tdme/engine/Rotation.h>
#include <tdme/engine/Rotations.h>
#include <tdme/math/Matrix4x4.h>
#include <tdme/math/Quaternion.h>
#include <tdme/math/Vector3.h>

using tdme::engine::Transformations;
using java::lang::Object;
using java::lang::String;
using java::lang::StringBuilder;
using tdme::engine::Rotation;
using tdme::engine::Rotations;
using tdme::math::Matrix4x4;
using tdme::math::Quaternion;
using tdme::math::Vector3;

Transformations::Transformations(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
	clinit();
}

Transformations::Transformations() 
	: Transformations(*static_cast< ::default_init_tag* >(0))
{
	ctor();
}

void Transformations::ctor()
{
	super::ctor();
	transformationsMatrix = (new Matrix4x4())->identity();
	translation = new Vector3();
	translationMatrix = new Matrix4x4();
	scale = new Vector3(1.0f, 1.0f, 1.0f);
	scaleMatrix = new Matrix4x4();
	rotations = new Rotations();
	rotationsQuaternionMatrix = new Matrix4x4();
	rotationsMatrix = (new Matrix4x4())->identity();
	rotationsPivot = new Vector3();
	rotationsTranslationsMatrix = (new Matrix4x4())->identity();
	pivot = new Vector3();
}

Vector3* Transformations::getTranslation()
{
	return translation;
}

Vector3* Transformations::getScale()
{
	return scale;
}

Vector3* Transformations::getPivot()
{
	return pivot;
}

Rotations* Transformations::getRotations()
{
	return rotations;
}

Matrix4x4* Transformations::getTransformationsMatrix()
{
	return transformationsMatrix;
}

void Transformations::fromTransformations(Transformations* transformations)
{
	translation->set(transformations->translation);
	scale->set(transformations->scale);
	pivot->set(transformations->getPivot());
	auto rotationIdx = 0;
	for (; rotationIdx < transformations->rotations->size(); rotationIdx++) {
		auto rotation = transformations->rotations->get(rotationIdx);
		auto _rotation = rotationIdx < rotations->size() ? rotations->get(rotationIdx) : static_cast< Rotation* >(nullptr);
		if (_rotation == nullptr) {
			_rotation = new Rotation();
			rotations->add(_rotation);
		}
		_rotation->fromRotation(rotation);
	}
	while (rotationIdx < rotations->size()) {
		rotations->remove(rotations->size() - 1);
	}
	translationMatrix->set(transformations->translationMatrix);
	scaleMatrix->set(transformations->scaleMatrix);
	transformationsMatrix->set(transformations->transformationsMatrix);
}

void Transformations::update()
{
	transformationsMatrix->identity();
	translationMatrix->identity()->translate(translation);
	scaleMatrix->identity()->scale(scale);
	rotations->update();
	rotationsMatrix->identity();
	rotationsPivot->set(pivot)->scale(-1.0f);
	rotationsMatrix->translate(rotationsPivot);
	rotations->quaternion->computeMatrix(rotationsQuaternionMatrix);
	rotationsMatrix->multiply(rotationsQuaternionMatrix);
	rotationsTranslationsMatrix->identity()->translate(pivot);
	rotationsMatrix->multiply(rotationsTranslationsMatrix);
	transformationsMatrix->multiply(scaleMatrix);
	transformationsMatrix->multiply(rotationsMatrix);
	transformationsMatrix->multiply(translationMatrix);
}

String* Transformations::toString()
{
	return ::java::lang::StringBuilder().append(u"Transformations [translation="_j)->append(static_cast< Object* >(translation))
		->append(u", translationMatrix="_j)
		->append(static_cast< Object* >(translationMatrix))
		->append(u", scale="_j)
		->append(static_cast< Object* >(scale))
		->append(u", scaleMatrix="_j)
		->append(static_cast< Object* >(scaleMatrix))
		->append(u", rotations="_j)
		->append(static_cast< Object* >(rotations))
		->append(u", transformationsMatrix="_j)
		->append(static_cast< Object* >(transformationsMatrix))
		->append(u"]"_j)->toString();
}

extern java::lang::Class* class_(const char16_t* c, int n);

java::lang::Class* Transformations::class_()
{
    static ::java::lang::Class* c = ::class_(u"tdme.engine.Transformations", 27);
    return c;
}

java::lang::Class* Transformations::getClass0()
{
	return class_();
}

