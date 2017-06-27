// Generated from /tdme/src/tdme/tests/GUITest.java
#include <tdme/tests/GUITest_init_1.h>

#include <java/lang/ClassCastException.h>
#include <java/lang/Exception.h>
#include <java/lang/Object.h>
#include <java/lang/String.h>
#include <java/lang/StringBuilder.h>
#include <tdme/gui/elements/GUITabController.h>
#include <tdme/gui/events/GUIActionListener_Type.h>
#include <tdme/gui/nodes/GUIElementNode.h>
#include <tdme/gui/nodes/GUINode.h>
#include <tdme/gui/nodes/GUINodeController.h>
#include <tdme/gui/nodes/GUIParentNode.h>
#include <tdme/gui/nodes/GUIScreenNode.h>
#include <tdme/tests/GUITest.h>
#include <tdme/utils/MutableString.h>
#include <tdme/utils/_Console.h>
#include <tdme/utils/_HashMap.h>

using tdme::tests::GUITest_init_1;
using java::lang::ClassCastException;
using java::lang::Exception;
using java::lang::Object;
using java::lang::String;
using java::lang::StringBuilder;
using tdme::gui::elements::GUITabController;
using tdme::gui::events::GUIActionListener_Type;
using tdme::gui::nodes::GUIElementNode;
using tdme::gui::nodes::GUINode;
using tdme::gui::nodes::GUINodeController;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::tests::GUITest;
using tdme::utils::MutableString;
using tdme::utils::_Console;
using tdme::utils::_HashMap;

template<typename T, typename U>
static T java_cast(U* u)
{
    if (!u) return static_cast<T>(nullptr);
    auto t = dynamic_cast<T>(u);
    if (!t) throw new ::java::lang::ClassCastException();
    return t;
}

GUITest_init_1::GUITest_init_1(GUITest *GUITest_this)
	: super(*static_cast< ::default_init_tag* >(0))
	, GUITest_this(GUITest_this)
{
	clinit();
	ctor();
}

void GUITest_init_1::onActionPerformed(GUIActionListener_Type* type, GUIElementNode* node)
{
	if (type == GUIActionListener_Type::PERFORMED && node->getName()->equals(u"button"_j)) {
		_Console::println(static_cast< Object* >(::java::lang::StringBuilder().append(node->getId())->append(u".actionPerformed()"_j)->toString()));
		auto values = new _HashMap();
		node->getScreenNode()->getValues(values);
		_Console::println(static_cast< Object* >(values));
		values->clear();
		values->put(u"select"_j, new MutableString(u"8"_j));
		values->put(u"input"_j, new MutableString(u"Enter some more text here!"_j));
		values->put(u"checkbox1"_j, new MutableString(u"1"_j));
		values->put(u"checkbox2"_j, new MutableString(u"1"_j));
		values->put(u"checkbox3"_j, new MutableString(u"1"_j));
		values->put(u"dropdown"_j, new MutableString(u"11"_j));
		values->put(u"radio"_j, new MutableString(u"3"_j));
		values->put(u"selectmultiple"_j, new MutableString(u"|1|2|3|15|16|17|"_j));
		node->getScreenNode()->setValues(values);
		(java_cast< GUITabController* >(node->getScreenNode()->getNodeById(u"tab1"_j)->getController()))->selectTab();
	} else if (type == GUIActionListener_Type::PERFORMED && node->getName()->equals(u"button2"_j)) {
		try {
{
				auto parentNode = java_cast< GUIParentNode* >((node->getScreenNode()->getNodeById(u"sadd_inner"_j)));
				parentNode->replaceSubNodes(::java::lang::StringBuilder().append(u"<dropdown-option text=\"Option 1\" value=\"1\" />"_j)->append(u"<dropdown-option text=\"Option 2\" value=\"2\" />"_j)
					->append(u"<dropdown-option text=\"Option 3\" value=\"3\" />"_j)
					->append(u"<dropdown-option text=\"Option 4\" value=\"4\" />"_j)
					->append(u"<dropdown-option text=\"Option 5\" value=\"5\" />"_j)
					->append(u"<dropdown-option text=\"Option 6\" value=\"6\" />"_j)
					->append(u"<dropdown-option text=\"Option 7\" value=\"7\" />"_j)
					->append(u"<dropdown-option text=\"Option 8\" value=\"8\" selected=\"true\" />"_j)
					->append(u"<dropdown-option text=\"Option 9\" value=\"9\" />"_j)
					->append(u"<dropdown-option text=\"Option 10\" value=\"10\" />"_j)->toString(), true);
			}

{
				auto parentNode = java_cast< GUIParentNode* >((node->getScreenNode()->getNodeById(u"sasb_inner"_j)));
				parentNode->replaceSubNodes(::java::lang::StringBuilder().append(u"<selectbox-option text=\"Option 1\" value=\"1\" />"_j)->append(u"<selectbox-option text=\"Option 2\" value=\"2\" />"_j)
					->append(u"<selectbox-option text=\"Option 3\" value=\"3\" />"_j)
					->append(u"<selectbox-option text=\"Option 4\" value=\"4\" selected=\"true\" />"_j)
					->append(u"<selectbox-option text=\"Option 5\" value=\"5\" />"_j)
					->append(u"<selectbox-option text=\"Option 6\" value=\"6\" />"_j)
					->append(u"<selectbox-option text=\"Option 7\" value=\"7\" />"_j)
					->append(u"<selectbox-option text=\"Option 8\" value=\"8\" />"_j)
					->append(u"<selectbox-option text=\"Option 9\" value=\"9\" />"_j)
					->append(u"<selectbox-option text=\"Option 10\" value=\"10\" />"_j)->toString(), true);
			}
		} catch (Exception* e) {
			e->printStackTrace();
		}
		(java_cast< GUITabController* >(node->getScreenNode()->getNodeById(u"tab2"_j)->getController()))->selectTab();
	}
}

extern java::lang::Class* class_(const char16_t* c, int n);

java::lang::Class* GUITest_init_1::class_()
{
    static ::java::lang::Class* c = ::class_(u"", 0);
    return c;
}

java::lang::Class* GUITest_init_1::getClass0()
{
	return class_();
}

