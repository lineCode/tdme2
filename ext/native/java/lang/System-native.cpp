// Generated from /Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Classes/classes.jar
#include <exception>

#include <sys/time.h>

#include <Array.h>
#include <ObjectArray.h>
#include <java/lang/System.h>

using java::lang::System;
extern void unimplemented_(const char16_t* name);

void System::arraycopy(char16_tArray* arg0, int32_t arg1, char16_tArray* arg2, int32_t arg3, int32_t arg4)
{ /* native */
	clinit();
	for (int i = 0; i < arg4; i++) arg2->set(i + arg3, arg0->get(i + arg1));
}

void System::arraycopy(floatArray* arg0, int32_t arg1, floatArray* arg2, int32_t arg3, int32_t arg4)
{ /* native */
	clinit();
	for (int i = 0; i < arg4; i++) arg2->set(i + arg3, arg0->get(i + arg1));
}

void System::arraycopy(ObjectArray* arg0, int32_t arg1, ObjectArray* arg2, int32_t arg3, int32_t arg4)
{ /* native */
	clinit();
	for (int i = 0; i < arg4; i++) arg2->set(i + arg3, arg0->get(i + arg1));
}

int64_t System::currentTimeMillis()
{ /* native */
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000ll) + (tv.tv_usec / 1000ll);
}

int32_t System::identityHashCode(Object* arg0)
{ /* native */
	clinit();
	unimplemented_(u"int32_t System::identityHashCode(Object* arg0)");
	return 0;
}

/* private: Properties* System::initProperties(Properties* arg0) */
String* System::mapLibraryName(String* arg0)
{ /* native */
	clinit();
	unimplemented_(u"String* System::mapLibraryName(String* arg0)");
	return 0;
}

int64_t System::nanoTime()
{ /* native */
	clinit();
	unimplemented_(u"int64_t System::nanoTime()");
	return 0;
}

/* private: void System::registerNatives() */
/* private: void System::setErr0(PrintStream* arg0) */
/* private: void System::setIn0(InputStream* arg0) */
/* private: void System::setOut0(PrintStream* arg0) */
