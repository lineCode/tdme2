#pragma once

#include <string>

#include <fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/gui/renderer/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/gui/nodes/GUINode.h>

using std::wstring;

using tdme::gui::nodes::GUINode;
using tdme::gui::nodes::GUIColor;
using tdme::gui::nodes::GUINode_Alignments;
using tdme::gui::nodes::GUINode_Border;
using tdme::gui::nodes::GUINode_Flow;
using tdme::gui::nodes::GUINode_Padding;
using tdme::gui::nodes::GUINode_RequestedConstraints;
using tdme::gui::nodes::GUINodeConditions;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::gui::renderer::GUIFont;
using tdme::gui::renderer::GUIRenderer;
using tdme::utils::MutableString;


struct default_init_tag;

/** 
 * GUI input internal node
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::nodes::GUIInputInternalNode final
	: public GUINode
{

public:
	typedef GUINode super;

	/** 
	 * Create max length
	 * @param s
	 * @return max length
	 */
	static int32_t createMaxLength(const wstring& s);

private:
	GUIFont* font {  };
	GUIColor* color {  };
	GUIColor* colorDisabled {  };
	MutableString* text {  };
	int32_t maxLength {  };
protected:

	/** 
	 * Constructor
	 * @param screen node
	 * @param parent mode
	 * @param id
	 * @param flow
	 * @param alignments
	 * @param requested constraints
	 * @param border
	 * @param padding
	 * @param show on
	 * @param hide on
	 * @param font
	 * @param color
	 * @param color if disabled
	 * @param text
	 * @throws Exception
	 */
	void ctor(GUIScreenNode* screenNode, GUIParentNode* parentNode, const wstring& id, GUINode_Flow* flow, GUINode_Alignments* alignments, GUINode_RequestedConstraints* requestedConstraints, GUIColor* backgroundColor, GUINode_Border* border, GUINode_Padding* padding, GUINodeConditions* showOn, GUINodeConditions* hideOn, const wstring& font, const wstring& color, const wstring& colorDisabled, MutableString* text, int32_t maxLength) /* throws(Exception) */;

public: /* protected */

	/** 
	 * @return node type
	 */
	const wstring getNodeType() override;
	bool isContentNode() override;

public:
	int32_t getContentWidth() override;
	int32_t getContentHeight() override;

	/** 
	 * @return font
	 */
	GUIFont* getFont();

	/** 
	 * @return text
	 */
	MutableString* getText();

	/** 
	 * @return max length
	 */
	int32_t getMaxLength();
	void dispose() override;
	void render(GUIRenderer* guiRenderer, vector<GUINode*>* floatingNodes) override;

	GUIInputInternalNode(GUIScreenNode* screenNode, GUIParentNode* parentNode, const wstring& id, GUINode_Flow* flow, GUINode_Alignments* alignments, GUINode_RequestedConstraints* requestedConstraints, GUIColor* backgroundColor, GUINode_Border* border, GUINode_Padding* padding, GUINodeConditions* showOn, GUINodeConditions* hideOn, const wstring& font, const wstring& color, const wstring& colorDisabled, MutableString* text, int32_t maxLength);
protected:
	GUIInputInternalNode(const ::default_init_tag&);
};
