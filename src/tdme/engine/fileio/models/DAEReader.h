// Generated from /tdme/src/tdme/engine/fileio/models/DAEReader.java

#pragma once

#include <fwd-tdme.h>
#include <java/lang/fwd-tdme.h>
#include <org/w3c/dom/fwd-tdme.h>
#include <tdme/engine/fileio/models/fwd-tdme.h>
#include <tdme/engine/model/fwd-tdme.h>
#include <tdme/tools/shared/model/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <java/lang/Object.h>

using java::lang::Object;
using java::lang::String;
using org::w3c::dom::Element;
using org::w3c::dom::Node;
using tdme::engine::fileio::models::DAEReader_AuthoringTool;
using tdme::engine::model::Color4;
using tdme::engine::model::Group;
using tdme::engine::model::Material;
using tdme::engine::model::Model_UpVector;
using tdme::engine::model::Model;
using tdme::tools::shared::model::LevelEditorLevel;
using tdme::utils::_ArrayList;
using tdme::utils::_HashMap;


struct default_init_tag;

/** 
 * Collada DAE reader
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::fileio::models::DAEReader final
	: public Object
{

public:
	typedef Object super;

private:
	static Color4* BLENDER_AMBIENT_NONE;
	static float BLENDER_AMBIENT_FROM_DIFFUSE_SCALE;
	static float BLENDER_DIFFUSE_SCALE;

public:

	/** 
	 * Reads Collada DAE file
	 * @param path name
	 * @param file name
	 * @throws Exception
	 * @return Model instance
	 */
	static Model* read(String* pathName, String* fileName) /* throws(Exception) */;

	/** 
	 * Reads Collada DAE file level
	 * @param path name
	 * @param file name
	 * @throws Exception
	 * @return Model instance
	 */
	static LevelEditorLevel* readLevel(String* pathName, String* fileName) /* throws(Exception) */;

private:

	/** 
	 * Get authoring tool
	 * @param xml root
	 * @return authoring tool
	 */
	static DAEReader_AuthoringTool* getAuthoringTool(Element* xmlRoot);

	/** 
	 * Get Up vector
	 * @param xml root
	 * @return up vector
	 * @throws ModelFileIOException
	 */
	static Model_UpVector* getUpVector(Element* xmlRoot) /* throws(ModelFileIOException) */;

	/** 
	 * Set up model import rotation matrix
	 * @param xml root
	 * @param model
	 */
	static void setupModelImportRotationMatrix(Element* xmlRoot, Model* model);

	/** 
	 * Set up model import scale matrix
	 * @param xml root
	 * @param model
	 */
	static void setupModelImportScaleMatrix(Element* xmlRoot, Model* model);

	/** 
	 * Read a DAE visual scene node
	 * @param authoring tool
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml node
	 * @param xml root
	 * @param frames per second
	 * @throws Exception
	 * @return group
	 */
	static Group* readVisualSceneNode(DAEReader_AuthoringTool* authoringTool, String* pathName, Model* model, Group* parentGroup, Element* xmlRoot, Element* xmlNode, float fps) /* throws(Exception) */;

	/** 
	 * Reads a DAE visual scene group node
	 * @param authoring tool
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml node
	 * @param xml root
	 * @param frames per seconds
	 * @throws Exception
	 * @return group
	 */
	static Group* readNode(DAEReader_AuthoringTool* authoringTool, String* pathName, Model* model, Group* parentGroup, Element* xmlRoot, Element* xmlNode, float fps) /* throws(Exception) */;

	/** 
	 * Reads a instance controller
	 * @param authoring tool
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml root
	 * @param xml node
	 * @return Group
	 * @throws Exception
	 */
	static Group* readVisualSceneInstanceController(DAEReader_AuthoringTool* authoringTool, String* pathName, Model* model, Group* parentGroup, Element* xmlRoot, Element* xmlNode) /* throws(Exception) */;

public:

	/** 
	 * Reads a geometry
	 * @param authoring tools
	 * @param path name
	 * @param model
	 * @param group
	 * @param xml root
	 * @param xml node id
	 * @param material symbols
	 * @throws Exception
	 */
	static void readGeometry(DAEReader_AuthoringTool* authoringTool, String* pathName, Model* model, Group* group, Element* xmlRoot, String* xmlNodeId, _HashMap* materialSymbols) /* throws(Exception) */;

	/** 
	 * Reads a material
	 * @param authoring tool
	 * @param path name
	 * @param model
	 * @param xml root
	 * @param xml node id
	 * @return material
	 * @throws Exception
	 */
	static Material* readMaterial(DAEReader_AuthoringTool* authoringTool, String* pathName, Model* model, Element* xmlRoot, String* xmlNodeId) /* throws(Exception) */;

private:

	/** 
	 * Determine displacement filename 
	 * @param path
	 * @param map type
	 * @param file name
	 * @return displacement file name or null
	 */
	static String* determineDisplacementFilename(String* path, String* mapType, String* fileName);

	/** 
	 * Make file name relative
	 * @param file name
	 * @return file name
	 */
	static String* makeFileNameRelative(String* fileName);

	/** 
	 * Get texture file name by id
	 * @param xml root
	 * @param xml texture id
	 * @return xml texture file name
	 */
	static String* getTextureFileNameById(Element* xmlRoot, String* xmlTextureId);

public:

	/** 
	 * Returns immediate children by tag names of parent
	 * @param parent
	 * @param name
	 * @return children with given name
	 */
	static _ArrayList* getChildrenByTagName(Element* parent, String* name);

private:

	/** 
	 * Converts an element to string
	 * @param node
	 * @return string representation
	 */
	static String* nodeToString(Node* node);

	// Generated

public:
	DAEReader();
protected:
	DAEReader(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();
	static void clinit();

private:
	virtual ::java::lang::Class* getClass0();
	friend class DAEReader_AuthoringTool;
	friend class DAEReader_determineDisplacementFilename_1;
};
