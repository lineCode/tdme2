#pragma once

#include <map>
#include <vector>

#include <tdme/gui/fwd-tdme.h>
#include <tdme/gui/elements/fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>

#include <tdme/gui/GUIParserException.h>

#include <ext/tinyxml/tinyxml.h>

using std::map;
using std::vector;

using tdme::gui::elements::GUIElement;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::gui::GUIParserException;
using tdme::ext::tinyxml::TiXmlElement;

/** 
 * GUI parser 
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::GUIParser final
{

private:
	static map<string, GUIElement*> elements;

public:

	/** 
	 * Parses a GUI XML file
	 * @param path name
	 * @param file name
	 * @return GUI screen node
	 * @throws IOException
	 */
	static GUIScreenNode* parse(const string& pathName, const string& fileName) throw (GUIParserException);

	/** 
	 * Parses a GUI XML content
	 * @param xml
	 * @return GUI screen node
	 * @throws IOException
	 */
	static GUIScreenNode* parse(const string& xml) throw (GUIParserException);

	/** 
	 * Parses a GUI XML file into parent node
	 * @param parent node
	 * @param path name
	 * @param file name
	 * @throws IOException
	 */
	static void parse(GUIParentNode* parentNode, const string& pathName, const string& fileName) throw (GUIParserException);

	/** 
	 * Parses a GUI XML content into parent node
	 * @param parent node
	 * @param xml
	 * @throws IOException
	 */
	static void parse(GUIParentNode* parentNode, const string& xml) throw (GUIParserException);

private:

	/** 
	 * Parse GUI node
	 * @param gui parent node
	 * @param xml parent node
	 * @param gui element
	 * @throws Exception
	 */
	static void parseGUINode(GUIParentNode* guiParentNode, TiXmlElement* xmlParentNode, GUIElement* guiElement) throw (GUIParserException);

	/** 
	 * Returns immediate children tags
	 * @param parent
	 * @param name
	 * @return children of parent node with given name
	 */
	static const vector<TiXmlElement*> getChildrenByTagName(TiXmlElement* parent, const char* name);

	/** 
	 * Get inner XML
	 * @param node
	 * @return string
	 */
	static const string getInnerXml(TiXmlElement* node);

	/**
	 * Parse template
	 * @param GUI parent node
	 * @param template xml node
	 * @param template
	 * @param attributes
	 * @param GUI element
	 */
	static void parseTemplate(GUIParentNode* guiParentNode, TiXmlElement* node, const string& _template, map<string, string>& attributes, GUIElement* guiElement);

public:

	/** 
	 * Unescape quotes
	 * @param string
	 * @return string with unescaped quotes
	 */
	static const string unescapeQuotes(const string& str);

	/**
	 * Escape quotes
	 * @param string
	 * @return string with escaped quotes
	 */
	static const string escapeQuotes(const string& str);

	/** 
	 * Add GUI element
	 * @param guiElement
	 * @throws GUIParserException
	 */
	static void addElement(GUIElement* guiElement) throw (GUIParserException);

	/**
	 * Initialize GUI elements
	 */
	static void initialize();
};
