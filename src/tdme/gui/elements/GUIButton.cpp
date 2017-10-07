#include <tdme/gui/elements/GUIButton.h>

#include <map>
#include <string>

#include <tdme/gui/elements/GUIButtonController.h>
#include <tdme/gui/nodes/GUIScreenNode.h>
#include <tdme/os/filesystem/_FileSystem.h>
#include <tdme/os/filesystem/_FileSystemException.h>
#include <tdme/os/filesystem/_FileSystemInterface.h>

using std::map;
using std::wstring;

using tdme::gui::elements::GUIButton;
using tdme::gui::elements::GUIButtonController;
using tdme::gui::nodes::GUIScreenNode;
using tdme::os::filesystem::_FileSystem;
using tdme::os::filesystem::_FileSystemException;
using tdme::os::filesystem::_FileSystemInterface;

wstring GUIButton::NAME = L"button";

GUIButton::GUIButton(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
}

GUIButton::GUIButton() throw (_FileSystemException)
	: GUIButton(*static_cast< ::default_init_tag* >(0))
{
	ctor();
}

void GUIButton::ctor() throw (_FileSystemException)
{
	template_ = _FileSystem::getInstance()->getContentAsString(L"resources/gui/definitions/elements", L"button.xml");
}

const wstring& GUIButton::getName()
{
	return NAME;
}

const wstring& GUIButton::getTemplate()
{
	return template_;
}

map<wstring, wstring>* GUIButton::getAttributes(GUIScreenNode* screenNode)
{
	attributes.clear();
	attributes[L"id"] = screenNode->allocateNodeId();
	attributes[L"width"] = L"auto";
	attributes[L"height"] = L"auto";
	attributes[L"text"] = L"";
	return &attributes;
}

GUINodeController* GUIButton::createController(GUINode* node)
{
	return new GUIButtonController(node);
}

