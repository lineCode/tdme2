#pragma once

#include <string>

#include <tdme/tdme.h>
#include <tdme/gui/elements/fwd-tdme.h>
#include <tdme/gui/events/fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/gui/nodes/GUIElementController.h>
#include <tdme/utils/fwd-tdme.h>

using std::string;

using tdme::gui::events::GUIKeyboardEvent;
using tdme::gui::events::GUIMouseEvent;
using tdme::gui::nodes::GUIElementController;
using tdme::gui::nodes::GUIInputInternalNode;
using tdme::gui::nodes::GUINode;
using tdme::utils::MutableString;

/** 
 * GUI input controller
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::elements::GUIInputController final
	: public GUIElementController
{
	friend class GUIInput;

private:
	static string CONDITION_DISABLED;
	static string CONDITION_ENABLED;
	GUIInputInternalNode* textInputNode { nullptr };
	bool disabled;

	/**
	 * Private constructor
	 * @param node node
	 */
	GUIInputController(GUINode* node);

public:
	bool isDisabled() override;
	void setDisabled(bool disabled) override;
	void initialize() override;
	void dispose() override;
	void postLayout() override;
	void handleMouseEvent(GUINode* node, GUIMouseEvent* event) override;
	void handleKeyboardEvent(GUINode* node, GUIKeyboardEvent* event) override;
	void tick() override;
	void onFocusGained() override;
	void onFocusLost() override;
	bool hasValue() override;
	const MutableString& getValue() override;
	void setValue(const MutableString& value) override;

};
