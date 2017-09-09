// Generated from /tdme/src/tdme/gui/elements/GUISelectBoxMultipleOptionController.java

#pragma once

#include <string>

#include <fwd-tdme.h>
#include <java/lang/fwd-tdme.h>
#include <tdme/gui/elements/fwd-tdme.h>
#include <tdme/gui/events/fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/gui/nodes/GUINodeController.h>

using std::wstring;

using tdme::gui::nodes::GUINodeController;
using tdme::gui::events::GUIKeyboardEvent;
using tdme::gui::events::GUIMouseEvent;
using tdme::gui::nodes::GUINode;
using tdme::gui::nodes::GUIParentNode;
using tdme::utils::MutableString;


struct default_init_tag;

/** 
 * GUI select box multiple option controller
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::elements::GUISelectBoxMultipleOptionController final
	: public GUINodeController
{

public:
	typedef GUINodeController super;

private:
	static wstring CONDITION_SELECTED;
	static wstring CONDITION_UNSELECTED;
	static wstring CONDITION_FOCUSSED;
	static wstring CONDITION_UNFOCUSSED;
	static wstring CONDITION_DISABLED;
	static wstring CONDITION_ENABLED;
	GUIParentNode* selectBoxMultipleNode {  };
	bool selected {  };
	bool focussed {  };
protected:

	/** 
	 * Constructor
	 * @param node
	 */
	void ctor(GUINode* node);

public:
	bool isDisabled() override;
	void setDisabled(bool disabled) override;

public: /* protected */

	/** 
	 * @return is selected
	 */
	bool isSelected();

	/** 
	 * Select
	 */
	void select();

	/** 
	 * Unselect
	 * @param checked
	 */
	void unselect();

	/** 
	 * Toggle selection
	 */
	void toggle();

public:

	/** 
	 * @return is focussed
	 */
	bool isFocussed();

public: /* protected */

	/** 
	 * Focus
	 */
	void focus();

	/** 
	 * Unfocus
	 * @param checked
	 */
	void unfocus();

public:
	void initialize() override;
	void dispose() override;
	void postLayout() override;
	void handleMouseEvent(GUINode* node, GUIMouseEvent* event) override;
	void handleKeyboardEvent(GUINode* node, GUIKeyboardEvent* event) override;
	void tick() override;
	void onFocusGained() override;
	void onFocusLost() override;
	bool hasValue() override;
	MutableString* getValue() override;
	void setValue(MutableString* value) override;

	// Generated

public: /* protected */
	GUISelectBoxMultipleOptionController(GUINode* node);
protected:
	GUISelectBoxMultipleOptionController(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();
	static void clinit();

private:
	virtual ::java::lang::Class* getClass0();
};
