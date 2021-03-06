#include <tdme/gui/elements/GUITab.h>

#include <map>
#include <string>

#include <tdme/gui/elements/GUITabController.h>
#include <tdme/gui/nodes/GUIScreenNode.h>
#include <tdme/os/filesystem/FileSystem.h>
#include <tdme/os/filesystem/FileSystemException.h>
#include <tdme/os/filesystem/FileSystemInterface.h>

using std::map;
using std::string;

using tdme::gui::elements::GUITab;
using tdme::gui::elements::GUITabController;
using tdme::gui::nodes::GUIScreenNode;
using tdme::os::filesystem::FileSystem;
using tdme::os::filesystem::FileSystemException;
using tdme::os::filesystem::FileSystemInterface;

string GUITab::NAME = "tab";

GUITab::GUITab()
{
	templateXML = FileSystem::getInstance()->getContentAsString("resources/gui-system/definitions/elements", "tab.xml");
}

const string& GUITab::getName()
{
	return NAME;
}

const string& GUITab::getTemplate()
{
	return templateXML;
}

map<string, string>& GUITab::getAttributes(GUIScreenNode* screenNode)
{
	attributes.clear();
	attributes["id"] = screenNode->allocateNodeId();
	return attributes;
}

GUINodeController* GUITab::createController(GUINode* node)
{
	return new GUITabController(node);
}

