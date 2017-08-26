// Generated from /tdme/src/tdme/os/_FileSystem.java

#pragma once

#include <tdme/os/fwd-tdme.h>

using tdme::os::_FileSystemInterface;

/** 
 * File system 
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::os::_FileSystem
{
private:
	static _FileSystemInterface* fileSystem;

public:

	/** 
	 * Singleton instance to retrieve file system
	 * will use standard file system by default if not set up different explicitly
	 * @return
	 */
	static _FileSystemInterface* getInstance();

	/** 
	 * Set up file system
	 * @param file system
	 */
	static void setupFileSystem(_FileSystemInterface* fileSystem);

};
