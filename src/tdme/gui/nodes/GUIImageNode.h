#pragma once

#include <string>

#include <tdme/tdme.h>
#include <tdme/engine/fileio/textures/fwd-tdme.h>
#include <tdme/gui/fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/gui/nodes/GUIColor.h>
#include <tdme/gui/renderer/fwd-tdme.h>
#include <tdme/gui/nodes/GUINode.h>
#include <tdme/gui/nodes/GUINode_Clipping.h>
#include <tdme/gui/nodes/GUINode_Scale9Grid.h>
#include <tdme/math/Matrix2D3x3.h>

using std::string;

using tdme::gui::nodes::GUINode;
using tdme::engine::fileio::textures::Texture;
using tdme::gui::nodes::GUIColor;
using tdme::gui::nodes::GUINode_Alignments;
using tdme::gui::nodes::GUINode_Border;
using tdme::gui::nodes::GUINode_Clipping;
using tdme::gui::nodes::GUINode_Flow;
using tdme::gui::nodes::GUINode_Padding;
using tdme::gui::nodes::GUINode_RequestedConstraints;
using tdme::gui::nodes::GUINode_Scale9Grid;
using tdme::gui::nodes::GUINodeConditions;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::gui::renderer::GUIRenderer;
using tdme::math::Matrix2D3x3;

/** 
 * GUI image node
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::nodes::GUIImageNode final
	: public GUINode
{
	friend class tdme::gui::GUIParser;

private:
	Texture* texture {  };
	int32_t textureId {  };
	GUIColor effectColorMul;
	GUIColor effectColorAdd;
	GUINode_Clipping clipping;

protected:
	/** 
	 * @return node type
	 */
	const string getNodeType() override;
	bool isContentNode() override;

	/**
	 * Constructor
	 * @param screen node
	 * @param parent node
	 * @param id
	 * @param flow
	 * @param alignments
	 * @param requested constraints
	 * @param background color
	 * @param background image
	 * @param background image scale 9 grid
	 * @param border
	 * @param padding
	 * @param show on
	 * @param hide on
	 * @param src
	 * @param effect color mul
	 * @param effect color add
	 */
	GUIImageNode(
		GUIScreenNode* screenNode,
		GUIParentNode* parentNode,
		const string& id,
		GUINode_Flow* flow,
		const GUINode_Alignments& alignments,
		const GUINode_RequestedConstraints& requestedConstraints,
		const GUIColor& backgroundColor,
		const string& backgroundImage,
		const GUINode_Scale9Grid& backgroundImageScale9Grid,
		const GUINode_Border& border,
		const GUINode_Padding& padding,
		const GUINodeConditions& showOn,
		const GUINodeConditions& hideOn,
		const string& source,
		const GUIColor& effectColorMul,
		const GUIColor& effectColorAdd,
		const GUINode_Scale9Grid& scale9Grid,
		const GUINode_Clipping& clipping
	) throw(GUIParserException);

public:
	int32_t getContentWidth() override;
	int32_t getContentHeight() override;
	void dispose() override;
	void render(GUIRenderer* guiRenderer, vector<GUINode*>& floatingNodes) override;

	/**
	 * Set image source
	 * @param source
	 */
	void setSource(const string source);

	/**
	 * @return image source
	 */
	const string& getSource();

	/**
	 * Set texture matrix
	 * @param texture matrix
	 */
	void setTextureMatrix(const Matrix2D3x3& textureMatrix);

	/**
	 * @return clipping
	 */
	GUINode_Clipping& getClipping();

	/**
	 * Create clipping
	 * @param all sides
	 * @param left
	 * @param top
	 * @param right
	 * @param bottom
	 */
	static GUINode_Clipping createClipping(const string& allClipping, const string& left, const string& top, const string& right, const string& bottom) throw (GUIParserException);

private:
	void init();

	string source;
	Matrix2D3x3 textureMatrix;
	GUINode_Scale9Grid scale9Grid;
};
