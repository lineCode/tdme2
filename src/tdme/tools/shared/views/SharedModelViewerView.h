// Generated from /tdme/src/tdme/tools/shared/views/SharedModelViewerView.java

#pragma once

#include <fwd-tdme.h>
#include <com/jogamp/opengl/fwd-tdme.h>
#include <java/io/fwd-tdme.h>
#include <java/lang/fwd-tdme.h>
#include <tdme/engine/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/tools/shared/controller/fwd-tdme.h>
#include <tdme/tools/shared/model/fwd-tdme.h>
#include <tdme/tools/shared/views/fwd-tdme.h>
#include <java/lang/Object.h>
#include <tdme/tools/shared/views/View.h>
#include <tdme/gui/events/GUIInputEventHandler.h>

using java::lang::Object;
using tdme::tools::shared::views::View;
using tdme::gui::events::GUIInputEventHandler;
using com::jogamp::opengl::GLAutoDrawable;
using java::io::File;
using java::lang::String;
using tdme::engine::Engine;
using tdme::math::Vector3;
using tdme::tools::shared::controller::ModelViewerScreenController;
using tdme::tools::shared::model::LevelEditorEntity;
using tdme::tools::shared::views::CameraRotationInputHandler;
using tdme::tools::shared::views::EntityBoundingVolumeView;
using tdme::tools::shared::views::EntityDisplayView;
using tdme::tools::shared::views::PopUps;


struct default_init_tag;

/** 
 * TDME Model Viewer View
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::tools::shared::views::SharedModelViewerView
	: public virtual Object
	, public virtual View
	, public virtual GUIInputEventHandler
{

public:
	typedef Object super;

public: /* protected */
	Engine* engine {  };

private:
	PopUps* popUps {  };
	ModelViewerScreenController* modelViewerScreenController {  };
	EntityDisplayView* entityDisplayView {  };
	EntityBoundingVolumeView* entityBoundingVolumeView {  };
	LevelEditorEntity* entity {  };
	bool loadModelRequested {  };
	bool initModelRequested {  };
	File* modelFile {  };
	CameraRotationInputHandler* cameraRotationInputHandler {  };
protected:

	/** 
	 * Public constructor
	 * @param pop ups view
	 */
	void ctor(PopUps* popUps);

public:

	/** 
	 * @return pop up views
	 */
	virtual PopUps* getPopUpsViews();

	/** 
	 * @return entity
	 */
	virtual LevelEditorEntity* getEntity();

	/** 
	 * Set entity
	 */
	virtual void setEntity(LevelEditorEntity* entity);

public: /* protected */

	/** 
	 * Init model
	 */
	virtual void initModel(GLAutoDrawable* drawable);

public:

	/** 
	 * @return current model file name
	 */
	virtual String* getFileName();

	/** 
	 * Issue file loading
	 */
	virtual void loadFile(String* pathName, String* fileName);

	/** 
	 * Triggers saving a map
	 */
	virtual void saveFile(String* pathName, String* fileName) /* throws(Exception) */;

	/** 
	 * Issue file reloading
	 */
	virtual void reloadFile();

	/** 
	 * Apply pivot
	 * @param x
	 * @param y
	 * @param z
	 */
	virtual void pivotApply(float x, float y, float z);
	void handleInputEvents() override;

	/** 
	 * Renders the scene 
	 */
	void display(GLAutoDrawable* drawable) override;

	/** 
	 * Init GUI elements
	 */
	virtual void updateGUIElements();

	/** 
	 * On init additional screens
	 * @param drawable
	 */
	virtual void onInitAdditionalScreens();

private:

	/** 
	 * Load settings
	 */
	void loadSettings();

public:
	void initialize() override;
	void activate() override;

private:

	/** 
	 * Store settings
	 */
	void storeSettings();

public:
	void deactivate() override;
	void dispose() override;

	/** 
	 * On load model
	 * @param oldModel
	 * @oaram entity
	 */
	virtual void onLoadModel(LevelEditorEntity* oldModel, LevelEditorEntity* model);

private:

	/** 
	 * Load a model
	 */
	void loadModel();

public: /* protected */

	/** 
	 * Load model
	 * @param name
	 * @param description
	 * @param path name
	 * @param file name
	 * @param pivot
	 * @return level editor entity
	 * @throws Exception
	 */
	virtual LevelEditorEntity* loadModel(String* name, String* description, String* pathName, String* fileName, Vector3* pivot) /* throws(Exception) */;

public:

	/** 
	 * On set entity data hook
	 */
	virtual void onSetEntityData();

	// Generated
	SharedModelViewerView(PopUps* popUps);
protected:
	SharedModelViewerView(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();

private:
	virtual ::java::lang::Class* getClass0();
};
