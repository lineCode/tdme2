// Generated from /tdme/src/tdme/tools/leveleditor/controller/TriggerScreenController.java

#pragma once

#include <tdme/tools/leveleditor/controller/fwd-tdme.h>
#include <tdme/tools/leveleditor/views/fwd-tdme.h>
#include <java/lang/Object.h>
#include <tdme/gui/events/Action.h>

using java::lang::Object;
using tdme::gui::events::Action;
using tdme::tools::leveleditor::controller::TriggerScreenController;
using tdme::tools::leveleditor::views::TriggerView;


struct default_init_tag;
class tdme::tools::leveleditor::controller::TriggerScreenController_TriggerScreenController_1
	: public virtual Object
	, public virtual Action
{

public:
	typedef Object super;
	void performAction() override;

	// Generated
	TriggerScreenController_TriggerScreenController_1(TriggerScreenController *TriggerScreenController_this, TriggerView* finalView);
	static ::java::lang::Class *class_();
	TriggerScreenController *TriggerScreenController_this;
	TriggerView* finalView;

private:
	virtual ::java::lang::Class* getClass0();
	friend class TriggerScreenController;
};
