// Generated from /tdme/src/tdme/tools/shared/files/LevelFileImport.java

#pragma once

#include <string>

#include <java/lang/fwd-tdme.h>
#include <tdme/tools/shared/files/fwd-tdme.h>
#include <tdme/tools/shared/model/fwd-tdme.h>
#include <tdme/engine/fileio/models/ModelFileIOException.h>
#include <tdme/os/filesystem/_FileSystemException.h>
#include <ext/jsonbox/JsonException.h>

using std::wstring;

using java::lang::String;
using tdme::engine::fileio::models::ModelFileIOException;
using tdme::tools::shared::model::LevelEditorLevel;
using tdme::os::filesystem::_FileSystemException;

using tdme::ext::jsonbox::JsonException;

/** 
 * TDME Level Editor File Export
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::tools::shared::files::LevelFileImport final
{

public:

	/** 
	 * Imports a level from a TDME level file to Level Editor
	 * @param game root
	 * @param path name
	 * @param file name
	 * @param level
	 * @throws file system exception
	 * @throws json exception
	 * @throws model file io exception
	 */
	static void doImport(const wstring& pathName, const wstring& fileName, LevelEditorLevel* level) throw (_FileSystemException, JsonException, ModelFileIOException);

	/** 
	 * Imports a level from a TDME level file to Level Editor
	 * @param path name
	 * @param file name
	 * @param level
	 * @param object id prefix
	 * @throws file system exception
	 * @throws json exception
	 * @throws model file io exception
	 */
	static void doImport(const wstring& pathName, const wstring& fileName, LevelEditorLevel* level, const wstring& objectIdPrefix) throw (_FileSystemException, JsonException, ModelFileIOException);
};
