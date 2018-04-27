
#pragma once

#include <map>
#include <vector>
#include <string>

#include <tdme/tdme.h>
#include <tdme/engine/fileio/models/fwd-tdme.h>
#include <tdme/engine/model/fwd-tdme.h>
#include <tdme/engine/model/Model.h>
#include <tdme/os/filesystem/fwd-tdme.h>
#include <tdme/tools/shared/model/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>

#include <tdme/engine/fileio/models/ModelFileIOException.h>
#include <tdme/os/filesystem/FileSystemException.h>

#include <ext/tinyxml/tinyxml.h>

using std::map;
using std::string;
using std::vector;

using tdme::engine::fileio::models::ModelFileIOException;
using tdme::engine::model::Color4;
using tdme::engine::model::Group;
using tdme::engine::model::Material;
using tdme::engine::model::UpVector;
using tdme::engine::model::Model;
using tdme::os::filesystem::FileSystemException;
using tdme::tools::shared::model::LevelEditorLevel;

using tdme::ext::tinyxml::TiXmlElement;

/** 
 * Collada DAE reader
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::engine::fileio::models::DAEReader final
{
	friend class DAEReader_determineDisplacementFilename_1;

private:
	static const Color4 BLENDER_AMBIENT_NONE;
	static float BLENDER_AMBIENT_FROM_DIFFUSE_SCALE;
	static float BLENDER_DIFFUSE_SCALE;

public:

	/** 
	 * Reads Collada DAE file
	 * @param path name
	 * @param file name
	 * @throws model file IO exception
	 * @throws file system exception
	 * @return model instance
	 */
	static Model* read(const string& pathName, const string& fileName) throw (ModelFileIOException, FileSystemException);

private:

	/** 
	 * Get authoring tool
	 * @param xml root
	 * @return authoring tool
	 */
	static Model::AuthoringTool getAuthoringTool(TiXmlElement* xmlRoot);

	/** 
	 * Get Up vector
	 * @param xml root
	 * @return up vector
	 * @throws ModelFileIOException
	 */
	static UpVector* getUpVector(TiXmlElement* xmlRoot) throw (ModelFileIOException);

	/** 
	 * Set up model import rotation matrix
	 * @param xml root
	 * @param model
	 */
	static void setupModelImportRotationMatrix(TiXmlElement* xmlRoot, Model* model);

	/** 
	 * Set up model import scale matrix
	 * @param xml root
	 * @param model
	 */
	static void setupModelImportScaleMatrix(TiXmlElement* xmlRoot, Model* model);

	/** 
	 * Read a DAE visual scene node
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml node
	 * @param xml root
	 * @param frames per second
	 * @return group
	 */
	static Group* readVisualSceneNode(const string& pathName, Model* model, Group* parentGroup, TiXmlElement* xmlRoot, TiXmlElement* xmlNode, float fps);

	/** 
	 * Reads a DAE visual scene group node
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml node
	 * @param xml root
	 * @param frames per seconds
	 * @throws model file IO exception
	 * @return group
	 */
	static Group* readNode(const string& pathName, Model* model, Group* parentGroup, TiXmlElement* xmlRoot, TiXmlElement* xmlNode, float fps) throw (ModelFileIOException);

	/** 
	 * Reads a instance controller
	 * @param path name
	 * @param model
	 * @param parent group
	 * @param xml root
	 * @param xml node
	 * @throws model file IO exception
	 * @return Group
	 * @throws Exception
	 */
	static Group* readVisualSceneInstanceController(const string& pathName, Model* model, Group* parentGroup, TiXmlElement* xmlRoot, TiXmlElement* xmlNode) throw (ModelFileIOException);

	/** 
	 * Reads a geometry
	 * @param path name
	 * @param model
	 * @param group
	 * @param xml root
	 * @param xml node id
	 * @param material symbols
	 * @throws model file IO exception
	 */
	static void readGeometry(const string& pathName, Model* model, Group* group, TiXmlElement* xmlRoot, const string& xmlNodeId, const map<string, string>* materialSymbols) throw (ModelFileIOException);

	/** 
	 * Reads a material
	 * @param path name
	 * @param model
	 * @param xml root
	 * @param xml node id
	 * @return material
	 */
	static Material* readMaterial(const string& pathName, Model* model, TiXmlElement* xmlRoot, const string& xmlNodeId);

	/** 
	 * Determine displacement filename 
	 * @param path
	 * @param map type
	 * @param file name
	 * @return displacement file name or null
	 */
	static const string determineDisplacementFilename(const string& path, const string& mapType, const string& fileName);

	/** 
	 * Make file name relative
	 * @param file name
	 * @return file name
	 */
	static const string makeFileNameRelative(const string& fileName);

	/** 
	 * Get texture file name by id
	 * @param xml root
	 * @param xml texture id
	 * @return xml texture file name
	 */
	static const string getTextureFileNameById(TiXmlElement* xmlRoot, const string& xmlTextureId);

public:

	/** 
	 * Returns immediate children tags by tag name
	 * @param parent
	 * @param name
	 * @return matching elements
	 */
	static const vector<TiXmlElement*> getChildrenByTagName(TiXmlElement* parent, const char* name);

	/**
	 * Returns immediate children tags
	 * @param parent
	 * @return elements
	 */
	static const vector<TiXmlElement*> getChildren(TiXmlElement* parent);
};
