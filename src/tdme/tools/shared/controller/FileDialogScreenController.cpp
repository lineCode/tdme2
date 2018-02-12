#include <tdme/tools/shared/controller/FileDialogScreenController.h>

#include <string>
#include <vector>


#include <tdme/gui/GUIParser.h>
#include <tdme/gui/events/Action.h>
#include <tdme/gui/events/GUIActionListener_Type.h>
#include <tdme/gui/nodes/GUIElementNode.h>
#include <tdme/gui/nodes/GUINode.h>
#include <tdme/gui/nodes/GUINodeController.h>
#include <tdme/gui/nodes/GUIParentNode.h>
#include <tdme/gui/nodes/GUIScreenNode.h>
#include <tdme/gui/nodes/GUITextNode.h>
#include <tdme/os/filesystem/FileSystem.h>
#include <tdme/os/filesystem/FileSystemInterface.h>
#include <tdme/tools/shared/controller/FileDialogScreenController_setupFileDialogListBox_1.h>
#include <tdme/utils/Console.h>
#include <tdme/utils/Exception.h>
#include <tdme/utils/MutableString.h>
#include <tdme/utils/StringUtils.h>

using std::vector;
using std::string;

using tdme::tools::shared::controller::FileDialogScreenController;
using tdme::gui::GUIParser;
using tdme::gui::events::Action;
using tdme::gui::events::GUIActionListener_Type;
using tdme::gui::nodes::GUIElementNode;
using tdme::gui::nodes::GUINode;
using tdme::gui::nodes::GUINodeController;
using tdme::gui::nodes::GUIParentNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::gui::nodes::GUITextNode;
using tdme::os::filesystem::FileSystem;
using tdme::os::filesystem::FileSystemInterface;
using tdme::tools::shared::controller::FileDialogScreenController_setupFileDialogListBox_1;
using tdme::utils::MutableString;
using tdme::utils::StringUtils;
using tdme::utils::Console;
using tdme::utils::Exception;

FileDialogScreenController::FileDialogScreenController() 
{
	this->cwd = FileSystem::getInstance()->getCurrentWorkingPathName();
	this->value = new MutableString();
	this->applyAction = nullptr;
}

FileDialogScreenController::~FileDialogScreenController() {
	if (applyAction != nullptr) delete applyAction;
}

GUIScreenNode* FileDialogScreenController::getScreenNode()
{
	return screenNode;
}

const string& FileDialogScreenController::getPathName()
{
	return cwd;
}

const string FileDialogScreenController::getFileName()
{
	return fileName->getController()->getValue().getString();
}

void FileDialogScreenController::initialize()
{
	try {
		screenNode = GUIParser::parse("resources/tools/shared/gui", "filedialog.xml");
		screenNode->setVisible(false);
		screenNode->addActionListener(this);
		screenNode->addChangeListener(this);
		caption = dynamic_cast< GUITextNode* >(screenNode->getNodeById("filedialog_caption"));
		files = dynamic_cast< GUIElementNode* >(screenNode->getNodeById("filedialog_files"));
		fileName = dynamic_cast< GUIElementNode* >(screenNode->getNodeById("filedialog_filename"));
	} catch (Exception& exception) {
		Console::print(string("FileDialogScreenController::initialize(): An error occurred: "));
		Console::println(string(exception.what()));
	}
}

void FileDialogScreenController::dispose()
{
}

void FileDialogScreenController::setupFileDialogListBox()
{
	auto directory = cwd;
	if (directory.length() > 50) {
		directory = "..." + StringUtils::substring(directory, directory.length() - 50 + 3);
	}

	caption->getText().set(captionText).append(directory);

	vector<string> fileList;
	try {
		auto directory = cwd;
		FileDialogScreenController_setupFileDialogListBox_1 extensionsFilter(this);
		FileSystem::getInstance()->list(directory, &fileList, &extensionsFilter);
	} catch (Exception& exception) {
		Console::print(string("FileDialogScreenController::setupFileDialogListBox(): An error occurred: "));
		Console::println(string(exception.what()));
	}

	auto filesInnerNode = dynamic_cast< GUIParentNode* >(files->getScreenNode()->getNodeById(files->getId() + "_inner"));
	auto idx = 1;
	string filesInnerNodeSubNodesXML = "";
	filesInnerNodeSubNodesXML =
		filesInnerNodeSubNodesXML +
		"<scrollarea width=\"100%\" height=\"100%\">\n";
	for (auto& file : fileList) {
		filesInnerNodeSubNodesXML =
			filesInnerNodeSubNodesXML +
			"<selectbox-option text=\"" +
			GUIParser::escapeQuotes(file) +
			"\" value=\"" +
			GUIParser::escapeQuotes(file) +
			"\" />\n";
	}
	filesInnerNodeSubNodesXML =
		filesInnerNodeSubNodesXML + "</scrollarea>\n";
	try {
		filesInnerNode->replaceSubNodes(filesInnerNodeSubNodesXML, true);
	} catch (Exception& exception) {
		Console::print(string("FileDialogScreenController::setupFileDialogListBox(): An error occurred: "));
		Console::println(string(exception.what()));
	}
}

void FileDialogScreenController::show(const string& cwd, const string& captionText, vector<string>* extensions, const string& fileName, Action* applyAction)
{
	try {
		this->cwd = FileSystem::getInstance()->getCanonicalPath(cwd, "");
	} catch (Exception& exception) {
		Console::print(string("FileDialogScreenController::show(): An error occurred: "));
		Console::println(string(exception.what()));
	}
	this->captionText = captionText;
	this->extensions = *extensions;
	this->fileName->getController()->setValue(value->set(fileName));
	setupFileDialogListBox();
	screenNode->setVisible(true);
	if (this->applyAction != nullptr) delete this->applyAction;
	this->applyAction = applyAction;
}

void FileDialogScreenController::close()
{
	screenNode->setVisible(false);
}

void FileDialogScreenController::onValueChanged(GUIElementNode* node)
{
	try {
		if (node->getId().compare(files->getId()) == 0) {
			auto selectedFile = node->getController()->getValue().getString();
			if (FileSystem::getInstance()->isPath(cwd + "/" + selectedFile) == true) {
				try {
					cwd = FileSystem::getInstance()->getCanonicalPath(cwd, selectedFile);
				} catch (Exception& exception) {
					Console::print(string("FileDialogScreenController::onValueChanged(): An error occurred: "));
					Console::println(string(exception.what()));
				}
				setupFileDialogListBox();
			} else {
				fileName->getController()->setValue(value->set(selectedFile));
			}
		}
	} catch (Exception& exception) {
		Console::print(string("FileDialogScreenController::onValueChanged(): An error occurred: "));
		Console::println(string(exception.what()));
	}
}

void FileDialogScreenController::onActionPerformed(GUIActionListener_Type* type, GUIElementNode* node)
{
	{
		auto v = type;
		if (v == GUIActionListener_Type::PERFORMED)
		{
			if (node->getId().compare("filedialog_apply") == 0) {
				if (applyAction != nullptr)
					applyAction->performAction();
			} else if (node->getId().compare("filedialog_abort") == 0) {
				close();
			}
		}
	}

}
