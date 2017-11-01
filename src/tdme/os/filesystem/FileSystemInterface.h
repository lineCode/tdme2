#pragma once

#include <vector>
#include <string>

#include <tdme/os/filesystem/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>

#include <tdme/os/filesystem/FileSystemException.h>

using std::vector;
using std::string;

using tdme::utils::FilenameFilter;

using tdme::os::filesystem::FileSystemException;

/** 
 * Interface to file system
 * @author Andreas Drewke
 * @version $Id$
 */
struct tdme::os::filesystem::FileSystemInterface
{
	/**
	 * Get file name
	 * @param path name
	 * @param file name
	 * @return complete filename with path and file
	 */
	virtual const string getFileName(const string& pathName, const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Get content as string
	 * @param path name
	 * @param file name
	 * @return string
	 */
	virtual const string getContentAsString(const string& pathName, const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Set content from string
	 * @param path name
	 * @param file name
	 * @return string
	 */
	virtual void setContentFromString(const string& pathName, const string& fileName, const string& content) throw (FileSystemException) = 0;

	/** 
	 * Get file content
	 * @param path name
	 * @param file name
	 * @return byte array
	 * @throws IOException
	 */
	virtual void getContent(const string& pathName, const string& fileName, vector<uint8_t>* content) throw (FileSystemException) = 0;

	/** 
	 * Get file content
	 * @param path name
	 * @param file name
	 * @param data
	 * @param length or -1 if data length should be used
	 * @throws IOException
	 */
	virtual void setContent(const string& pathName, const string& fileName, vector<uint8_t>* content) throw (FileSystemException) = 0;

	/**
	 * Get file content as string array
	 * @param path name
	 * @param file name
	 * @return byte array
	 * @throws IOException
	 */
	virtual void getContentAsStringArray(const string& pathName, const string& fileName, vector<string>* content) throw (FileSystemException) = 0;

	/**
	 * Set file content as string array
	 * @param path name
	 * @param file name
	 * @param string array
	 * @return byte array
	 * @throws IOException
	 */
	virtual void setContentFromStringArray(const string& pathName, const string& fileName, vector<string>* content) throw (FileSystemException) = 0;

	/**
	 * List files for given path and filter by a file name filter if not null 
	 * @param path name
	 * @param filter or null
	 * @return file names 
	 */
	virtual void list(const string& pathName, vector<string>* files, FilenameFilter* filter = nullptr) throw (FileSystemException) = 0;

	/**
	 * Check if file is a path
	 * @param path name
	 * @return if file is a path
	 */
	virtual bool isPath(const string& pathName) throw (FileSystemException) = 0;

	/**
	 * Check if file exists
	 * @param file name
	 * @return bool if file exists
	 */
	virtual bool fileExists(const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Get canonical path name
	 * @param path name
	 * @param file name
	 * @return canonical path
	 */
	virtual const string getCanonicalPath(const string& pathName, const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Get current working path name
	 * @return current working path
	 */
	virtual const string getCurrentWorkingPathName() throw (FileSystemException) = 0;

	/**
	 * Get path name
	 * @param file name
	 * @return canonical path
	 */
	virtual const string getPathName(const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Get file name
	 * @param file name
	 * @return canonical path
	 */
	virtual const string getFileName(const string& fileName) throw (FileSystemException) = 0;

	/**
	 * Create path
	 * @param path name
	 */
	virtual void createPath(const string& pathName) throw (FileSystemException) = 0;

	/**
	 * Remove path
	 * @param path name
	 * @return success
	 */
	virtual void removePath(const string& pathName) throw (FileSystemException) = 0;

	/**
	 * Remove file
	 * @param path name
	 * @param file name
	 * @return success
	 */
	virtual void removeFile(const string& pathName, const string& fileName) throw (FileSystemException) = 0;
};