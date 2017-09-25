// Generated from /tdme/src/tdme/gui/elements/GUIScrollArea.java
#include <tdme/gui/elements/GUIScrollArea.h>

#include <tdme/gui/elements/GUIScrollAreaController.h>
#include <tdme/gui/nodes/GUIScreenNode.h>
#include <tdme/os/_FileSystem.h>
#include <tdme/os/_FileSystemException.h>
#include <tdme/os/_FileSystemInterface.h>

using tdme::gui::elements::GUIScrollArea;
using tdme::gui::elements::GUIScrollAreaController;
using tdme::gui::nodes::GUIScreenNode;
using tdme::os::_FileSystem;
using tdme::os::_FileSystemException;
using tdme::os::_FileSystemInterface;

GUIScrollArea::GUIScrollArea(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
	clinit();
}

GUIScrollArea::GUIScrollArea() throw (_FileSystemException)
	: GUIScrollArea(*static_cast< ::default_init_tag* >(0))
{
	ctor();
}

wstring GUIScrollArea::NAME = L"scrollarea";

void GUIScrollArea::ctor() throw (_FileSystemException)
{
	super::ctor();
	template_ = _FileSystem::getInstance()->getContentAsString(L"resources/gui/definitions/elements", L"scrollarea.xml");
}

const wstring& GUIScrollArea::getName()
{
	return NAME;
}

const wstring& GUIScrollArea::getTemplate()
{
	return template_;
}

map<wstring, wstring>* GUIScrollArea::getAttributes(GUIScreenNode* screenNode)
{
	attributes.clear();
	attributes[L"id"] = screenNode->allocateNodeId();
	attributes[L"width"] = L"100%";
	attributes[L"height"] = L"100%";
	attributes[L"horizontal-align"] = L"left";
	attributes[L"vertical-align"] = L"top";
	attributes[L"alignment"] = L"vertical";
	attributes[L"background-color"] = L"transparent";
	attributes[L"show-on"] = L"";
	attributes[L"hide-on"] = L"";
	return &attributes;
}

GUINodeController* GUIScrollArea::createController(GUINode* node)
{
	return new GUIScrollAreaController(node);
}

extern java::lang::Class* class_(const char16_t* c, int n);

java::lang::Class* GUIScrollArea::class_()
{
    static ::java::lang::Class* c = ::class_(u"tdme.gui.elements.GUIScrollArea", 31);
    return c;
}

java::lang::Class* GUIScrollArea::getClass0()
{
	return class_();
}

