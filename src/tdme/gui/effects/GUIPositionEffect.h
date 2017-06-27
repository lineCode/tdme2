// Generated from /tdme/src/tdme/gui/effects/GUIPositionEffect.java

#pragma once

#include <fwd-tdme.h>
#include <tdme/gui/effects/fwd-tdme.h>
#include <tdme/gui/renderer/fwd-tdme.h>
#include <tdme/gui/effects/GUIEffect.h>

using tdme::gui::effects::GUIEffect;
using tdme::gui::renderer::GUIRenderer;


struct default_init_tag;

/** 
 * GUI position effect
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::effects::GUIPositionEffect
	: public GUIEffect
{

public:
	typedef GUIEffect super;

private:
	float positionXStart {  };
	float positionXEnd {  };
	float positionYStart {  };
	float positionYEnd {  };
	float positionX {  };
	float positionY {  };
protected:

	/** 
	 * Public constructor
	 */
	void ctor();

public:

	/** 
	 * @return position X start
	 */
	virtual float getPositionXStart();

	/** 
	 * Set position X start
	 * @param position X start
	 */
	virtual void setPositionXStart(float positionXStart);

	/** 
	 * @return position X end
	 */
	virtual float getPositionXEnd();

	/** 
	 * Set position X end
	 * @param position X end
	 */
	virtual void setPositionXEnd(float positionXEnd);

	/** 
	 * @return position Y start
	 */
	virtual float getPositionYStart();

	/** 
	 * Set position Y start
	 * @param position Y start
	 */
	virtual void setPositionYStart(float positionYStart);

	/** 
	 * @return get position Y end
	 */
	virtual float getPositionYEnd();

	/** 
	 * Set position Y end
	 * @param position Y end
	 */
	virtual void setPositionYEnd(float positionYEnd);
	void apply(GUIRenderer* guiRenderer) override;

	// Generated
	GUIPositionEffect();
protected:
	GUIPositionEffect(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();

private:
	void init();
	virtual ::java::lang::Class* getClass0();
};
