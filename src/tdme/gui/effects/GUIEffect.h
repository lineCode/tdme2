// Generated from /tdme/src/tdme/gui/effects/GUIEffect.java

#pragma once

#include <fwd-tdme.h>
#include <tdme/gui/effects/fwd-tdme.h>
#include <tdme/gui/events/fwd-tdme.h>
#include <tdme/gui/renderer/fwd-tdme.h>
#include <java/lang/Object.h>

using java::lang::Object;
using tdme::gui::events::Action;
using tdme::gui::renderer::GUIRenderer;


struct default_init_tag;

/** 
 * GUI Effect base class
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::gui::effects::GUIEffect
	: public virtual Object
{

public:
	typedef Object super;

public: /* protected */
	bool active {  };
	float timeTotal {  };
	float timeLeft {  };
	float timePassed {  };
	Action* action {  };
protected:

	/** 
	 * Public constructor
	 */
	void ctor();

public:

	/** 
	 * @return active
	 */
	virtual bool isActive();

	/** 
	 * @return time total
	 */
	virtual float getTimeTotal();

	/** 
	 * Set time total
	 * @param time total
	 */
	virtual void setTimeTotal(float timeTotal);

	/** 
	 * @return action to be performed on effect end
	 */
	virtual Action* getAction();

	/** 
	 * Set action to be performed on effect end
	 * @param action
	 */
	virtual void setAction(Action* action);

	/** 
	 * Start this effect
	 */
	virtual void start();

	/** 
	 * Updates the effect to GUI renderer and updates time
	 * @param gui renderer
	 */
	virtual void update(GUIRenderer* guiRenderer);

	/** 
	 * Apply effect
	 * @param GUI renderer
	 */
	virtual void apply(GUIRenderer* guiRenderer) = 0;

	// Generated
	GUIEffect();
protected:
	GUIEffect(const ::default_init_tag&);


public:
	static ::java::lang::Class *class_();

private:
	virtual ::java::lang::Class* getClass0();
};
