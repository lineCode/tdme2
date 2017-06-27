// Generated from /tdme/src/tdme/engine/fileio/models/DAEReader.java
#include <tdme/engine/fileio/models/DAEReader_AuthoringTool.h>

#include <java/io/Serializable.h>
#include <java/lang/ArrayStoreException.h>
#include <java/lang/Comparable.h>
#include <java/lang/Enum.h>
#include <java/lang/IllegalArgumentException.h>
#include <java/lang/String.h>
#include <SubArray.h>
#include <ObjectArray.h>

using tdme::engine::fileio::models::DAEReader_AuthoringTool;
using java::io::Serializable;
using java::lang::ArrayStoreException;
using java::lang::Comparable;
using java::lang::Enum;
using java::lang::IllegalArgumentException;
using java::lang::String;

template<typename ComponentType, typename... Bases> struct SubArray;
namespace java {
namespace io {
typedef ::SubArray< ::java::io::Serializable, ::java::lang::ObjectArray > SerializableArray;
}  // namespace io

namespace lang {
typedef ::SubArray< ::java::lang::Comparable, ObjectArray > ComparableArray;
typedef ::SubArray< ::java::lang::Enum, ObjectArray, ComparableArray, ::java::io::SerializableArray > EnumArray;
}  // namespace lang
}  // namespace java

namespace tdme {
namespace engine {
namespace fileio {
namespace models {
typedef ::SubArray< ::tdme::engine::fileio::models::DAEReader_AuthoringTool, ::java::lang::EnumArray > DAEReader_AuthoringToolArray;
}  // namespace models
}  // namespace fileio
}  // namespace engine
}  // namespace tdme

DAEReader_AuthoringTool::DAEReader_AuthoringTool(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
	clinit();
}

DAEReader_AuthoringTool::DAEReader_AuthoringTool(::java::lang::String* name, int ordinal)
	: DAEReader_AuthoringTool(*static_cast< ::default_init_tag* >(0))
{
	ctor(name, ordinal);
}

DAEReader_AuthoringTool* tdme::engine::fileio::models::DAEReader_AuthoringTool::UNKNOWN = new DAEReader_AuthoringTool(u"UNKNOWN"_j, 0);
DAEReader_AuthoringTool* tdme::engine::fileio::models::DAEReader_AuthoringTool::BLENDER = new DAEReader_AuthoringTool(u"BLENDER"_j, 1);
extern java::lang::Class* class_(const char16_t* c, int n);

java::lang::Class* DAEReader_AuthoringTool::class_()
{
    static ::java::lang::Class* c = ::class_(u"tdme.engine.fileio.models.DAEReader.AuthoringTool", 49);
    return c;
}

DAEReader_AuthoringTool* DAEReader_AuthoringTool::valueOf(String* a0)
{
	if (BLENDER->toString()->equals(a0))
		return BLENDER;
	if (UNKNOWN->toString()->equals(a0))
		return UNKNOWN;
	throw new IllegalArgumentException(a0);
}

DAEReader_AuthoringToolArray* DAEReader_AuthoringTool::values()
{
	return new DAEReader_AuthoringToolArray({
		BLENDER,
		UNKNOWN,
	});
}

java::lang::Class* DAEReader_AuthoringTool::getClass0()
{
	return class_();
}

