#include <tdme/gui/elements/GUIScrollAreaVerticalController.h>

#include <tdme/gui/elements/GUIScrollAreaVerticalController_initialize_1.h>
#include <tdme/gui/nodes/GUIElementNode.h>
#include <tdme/gui/nodes/GUINode.h>
#include <tdme/gui/nodes/GUIParentNode.h>
#include <tdme/gui/nodes/GUIScreenNode.h>

using tdme::gui::elements::GUIScrollAreaVerticalController;
using tdme::gui::elements::GUIScrollAreaVerticalController_initialize_1;
using tdme::gui::nodes::GUIElementNode;
using tdme::gui::nodes::GUINode;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;

GUIScrollAreaVerticalController::GUIScrollAreaVerticalController(const ::default_init_tag&)
	: super(*static_cast< ::default_init_tag* >(0))
{
}

GUIScrollAreaVerticalController::GUIScrollAreaVerticalController(GUINode* node) 
	: GUIScrollAreaVerticalController(*static_cast< ::default_init_tag* >(0))
{
	ctor(node);
}

void GUIScrollAreaVerticalController::ctor(GUINode* node)
{
	super::ctor(node);
}

bool GUIScrollAreaVerticalController::isDisabled()
{
	return false;
}

void GUIScrollAreaVerticalController::setDisabled(bool disabled)
{
}

void GUIScrollAreaVerticalController::initialize()
{
	auto const contentNode = dynamic_cast< GUIParentNode* >(node->getScreenNode()->getNodeById(node->getId() + L"_inner"));
	auto const upArrowNode = dynamic_cast< GUIElementNode* >(node->getScreenNode()->getNodeById(node->getId() + L"_scrollbar_vertical_layout_up"));
	auto const downArrowNode = dynamic_cast< GUIElementNode* >(node->getScreenNode()->getNodeById(node->getId() + L"_scrollbar_vertical_layout_down"));
	node->getScreenNode()->addActionListener(new GUIScrollAreaVerticalController_initialize_1(this, upArrowNode, contentNode, downArrowNode));
}

void GUIScrollAreaVerticalController::dispose()
{
}

void GUIScrollAreaVerticalController::postLayout()
{
}

void GUIScrollAreaVerticalController::handleMouseEvent(GUINode* node, GUIMouseEvent* event)
{
}

void GUIScrollAreaVerticalController::handleKeyboardEvent(GUINode* node, GUIKeyboardEvent* event)
{
}

void GUIScrollAreaVerticalController::tick()
{
}

void GUIScrollAreaVerticalController::onFocusGained()
{
}

void GUIScrollAreaVerticalController::onFocusLost()
{
}

bool GUIScrollAreaVerticalController::hasValue()
{
	return false;
}

MutableString* GUIScrollAreaVerticalController::getValue()
{
	return nullptr;
}

void GUIScrollAreaVerticalController::setValue(MutableString* value)
{
}

