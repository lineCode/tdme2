
#pragma once

#include <string>

#include <tdme/tdme.h>
#include <tdme/gui/events/fwd-tdme.h>
#include <tdme/gui/nodes/fwd-tdme.h>
#include <tdme/math/fwd-tdme.h>
#include <tdme/tools/shared/controller/fwd-tdme.h>
#include <tdme/tools/shared/model/fwd-tdme.h>
#include <tdme/tools/shared/views/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <tdme/tools/shared/controller/ScreenController.h>
#include <tdme/gui/events/GUIActionListener.h>
#include <tdme/gui/events/GUIChangeListener.h>

using std::string;

using tdme::tools::shared::controller::ScreenController;
using tdme::gui::events::GUIActionListener;
using tdme::gui::events::GUIChangeListener;
using tdme::gui::events::GUIActionListener_Type;
using tdme::gui::nodes::GUIElementNode;
using tdme::gui::nodes::GUIScreenNode;
using tdme::gui::nodes::GUITextNode;
using tdme::math::Vector3;
using tdme::tools::shared::controller::EntityBaseSubScreenController;
using tdme::tools::shared::controller::EntityBoundingVolumeSubScreenController;
using tdme::tools::shared::controller::EntityDisplaySubScreenController;
using tdme::tools::shared::controller::FileDialogPath;
using tdme::tools::shared::model::LevelEditorEntity;
using tdme::tools::shared::views::SharedModelEditorView;
using tdme::utils::MutableString;

/** 
 * Model editor screen controller
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::tools::shared::controller::ModelEditorScreenController final
	: public ScreenController
	, public GUIActionListener
	, public GUIChangeListener
{
	friend class ModelEditorScreenController_ModelEditorScreenController_1;
	friend class ModelEditorScreenController_onModelLoad_2;
	friend class ModelEditorScreenController_onModelSave_3;

private:
	static MutableString* TEXT_EMPTY;
	EntityBaseSubScreenController* entityBaseSubScreenController {  };
	EntityDisplaySubScreenController* entityDisplaySubScreenController {  };
	EntityBoundingVolumeSubScreenController* entityBoundingVolumeSubScreenController {  };
	SharedModelEditorView* view {  };
	GUIScreenNode* screenNode {  };
	GUITextNode* screenCaption {  };
	GUIElementNode* modelReload {  };
	GUIElementNode* modelSave {  };
	GUIElementNode* pivotX {  };
	GUIElementNode* pivotY {  };
	GUIElementNode* pivotZ {  };
	GUIElementNode* pivotApply {  };
	GUIElementNode* statsOpaqueFaces {  };
	GUIElementNode* statsTransparentFaces {  };
	GUIElementNode* statsMaterialCount {  };
	GUIElementNode* renderingDynamicShadowing {  };
	GUIElementNode* renderingApply {  };
	GUIElementNode* materialsDropdown {  };
	GUIElementNode* materialsDropdownApply {  };
	GUIElementNode* materialsMaterialName {  };
	GUIElementNode* materialsMaterialAmbient {  };
	GUIElementNode* materialsMaterialDiffuse {  };
	GUIElementNode* materialsMaterialSpecular {  };
	GUIElementNode* materialsMaterialEmission {  };
	GUIElementNode* materialsMaterialShininess {  };
	GUIElementNode* materialsMaterialDiffuseTexture {  };
	GUIElementNode* materialsMaterialDiffuseTransparencyTexture {  };
	GUIElementNode* materialsMaterialNormalTexture {  };
	GUIElementNode* materialsMaterialSpecularTexture {  };
	GUIElementNode* materialsMaterialDiffuseTextureLoad {  };
	GUIElementNode* materialsMaterialDiffuseTransparencyTextureLoad {  };
	GUIElementNode* materialsMaterialNormalTextureLoad {  };
	GUIElementNode* materialsMaterialSpecularTextureLoad {  };
	GUIElementNode* materialsMaterialDiffuseTextureClear {  };
	GUIElementNode* materialsMaterialDiffuseTransparencyTextureClear {  };
	GUIElementNode* materialsMaterialNormalTextureClear {  };
	GUIElementNode* materialsMaterialSpecularTextureClear {  };
	GUIElementNode* materialsMaterialUseMaskedTransparency {  };
	GUIElementNode* materialsMaterialApply {  };
	GUIElementNode* animationsDropDown {  };
	GUIElementNode* animationsDropDownApply {  };
	GUIElementNode* animationsDropDownDelete {  };
	GUIElementNode* animationsAnimationStartFrame {  };
	GUIElementNode* animationsAnimationEndFrame {  };
	GUIElementNode* animationsAnimationOverlayFromGroupIdDropDown {  };
	GUIElementNode* animationsAnimationLoop {  };
	GUIElementNode* animationsAnimationName {  };
	GUIElementNode* animationsAnimationApply {  };

	MutableString* value {  };
	FileDialogPath* modelPath {  };

public:

	/**
	 * Get view
	 */
	SharedModelEditorView* getView();

	/** 
	 * @return entity display sub screen controller
	 */
	EntityDisplaySubScreenController* getEntityDisplaySubScreenController();

	/** 
	 * @return entity bounding volume sub screen controller
	 */
	EntityBoundingVolumeSubScreenController* getEntityBoundingVolumeSubScreenController();
	GUIScreenNode* getScreenNode() override;

	/** 
	 * @return model path
	 */
	FileDialogPath* getModelPath();
	void initialize() override;
	void dispose() override;

	/** 
	 * Set screen caption
	 * @param text
	 */
	void setScreenCaption(const string& text);

	/** 
	 * Set up general entity data
	 * @param name
	 * @param description
	 */
	void setEntityData(const string& name, const string& description);

	/** 
	 * Unset entity data
	 */
	void unsetEntityData();

	/** 
	 * Set up entity properties
	 * @param preset id
	 * @param entity properties
	 * @param selected name
	 */
	void setEntityProperties(const string& presetId, LevelEditorEntity* entity, const string& selectedName);

	/** 
	 * Unset entity properties
	 */
	void unsetEntityProperties();

	/** 
	 * Set pivot tab
	 * @param pivot
	 */
	void setPivot(const Vector3& pivot);

	/** 
	 * Unset pivot tab
	 */
	void unsetPivot();

	/**
	 * Set renering options
	 * @param entity
	 */
	void setRendering(LevelEditorEntity* entity);

	/**
	 * Unset rendering
	 */
	void unsetRendering();

	/**
	 * Set materials
	 * @param entity
	 */
	void setMaterials(LevelEditorEntity* entity);

	/**
	 * Unset materials
	 */
	void unsetMaterials();

	/**
	 * On material drop down apply
	 */
	void onMaterialDropDownApply();

	/**
	 * On material apply
	 */
	void onMaterialApply();

	/**
	 * On material load diffuse texture
	 */
	void onMaterialLoadDiffuseTexture();

	/**
	 * On material load diffuse transparency texture
	 */
	void onMaterialLoadDiffuseTransparencyTexture();

	/**
	 * On material load normal texture
	 */
	void onMaterialLoadNormalTexture();

	/**
	 * On material load specular texture
	 */
	void onMaterialLoadSpecularTexture();

	/**
	 * On material clear texture
	 */
	void onMaterialClearTexture(GUIElementNode* guiElementNode);

	/**
	 * Set animations
	 */
	void setAnimations(LevelEditorEntity* entity);

	/**
	 * On animation drop down value changed
	 */
	void onAnimationDropDownValueChanged();

	/**
	 * On animation drop down apply
	 */
	void onAnimationDropDownApply();

	/**
	 * On animation drop down delete
	 */
	void onAnimationDropDownDelete();

	/**
	 * On animation apply
	 */
	void onAnimationApply();

	/**
	 * Unset animations
	 */
	void unsetAnimations();

	/** 
	 * Set up model statistics
	 * @param stats opaque faces
	 * @param stats transparent faces
	 * @param stats material count
	 */
	void setStatistics(int32_t statsOpaqueFaces, int32_t statsTransparentFaces, int32_t statsMaterialCount);

	/** 
	 * On quit
	 */
	void onQuit();

	/** 
	 * On model load
	 */
	void onModelLoad();

	/** 
	 * On model save
	 */
	void onModelSave();

	/** 
	 * On model reload
	 */
	void onModelReload();

	/** 
	 * On pivot apply
	 */
	void onPivotApply();

	/**
	 * On rendering apply
	 */
	void onRenderingApply();

	/**
	 * Save file
	 * @param path name
	 * @param file name
	 */
	void saveFile(const string& pathName, const string& fileName) /* throws(Exception) */;

	/**
	 * Load file
	 * @param path name
	 * @param file name
	 */
	void loadFile(const string& pathName, const string& fileName) /* throws(Exception) */;

	/** 
	 * Shows the error pop up
	 */
	void showErrorPopUp(const string& caption, const string& message);

	/**
	 * On value changed
	 * @param node
	 */
	void onValueChanged(GUIElementNode* node) override;

	/**
	 * On action performed
	 * @param type
	 * @param node
	 */
	void onActionPerformed(GUIActionListener_Type* type, GUIElementNode* node) override;

	/**
	 * Public constructor
	 * @param view
	 */
	ModelEditorScreenController(SharedModelEditorView* view);

	/**
	 * Destructor
	 */
	virtual ~ModelEditorScreenController();
};