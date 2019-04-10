#pragma once

#if defined(VULKAN)
	#define GLFW_INCLUDE_VULKAN
	#include <GLFW/glfw3.h>

	#define KEYBOARD_MODIFIER_SHIFT	GLFW_MOD_SHIFT
	#define KEYBOARD_MODIFIER_CTRL GLFW_MOD_CONTROL
	#define KEYBOARD_MODIFIER_ALT GLFW_MOD_ALT

	#define MOUSE_BUTTON_DOWN GLFW_PRESS
	#define MOUSE_BUTTON_UP GLFW_RELEASE

	#define KEYBOARD_KEYCODE_TAB GLFW_KEY_TAB
	#define KEYBOARD_KEYCODE_TAB_SHIFT -2
	#define KEYBOARD_KEYCODE_BACKSPACE GLFW_KEY_BACKSPACE
	#define KEYBOARD_KEYCODE_RETURN GLFW_KEY_ENTER
	#define KEYBOARD_KEYCODE_DELETE GLFW_KEY_DELETE
	#define KEYBOARD_KEYCODE_SPACE GLFW_KEY_SPACE
	#define KEYBOARD_KEYCODE_LEFT GLFW_KEY_LEFT
	#define KEYBOARD_KEYCODE_UP GLFW_KEY_UP
	#define KEYBOARD_KEYCODE_RIGHT GLFW_KEY_RIGHT
	#define KEYBOARD_KEYCODE_DOWN GLFW_KEY_DOWN
	#define KEYBOARD_KEYCODE_POS1 GLFW_KEY_HOME
	#define KEYBOARD_KEYCODE_END GLFW_KEY_END
	#define KEYBOARD_KEYCODE_ESCAPE GLFW_KEY_ESCAPE

#else
	#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__linux__) || defined(_WIN32)
		#include <GL/freeglut.h>
	#elif defined(__APPLE__)
		#include <GLUT/glut.h>
	#elif defined(__HAIKU__)
		#include <GL/glut.h>
	#endif

	#define KEYBOARD_MODIFIER_SHIFT	GLUT_ACTIVE_SHIFT
	#define KEYBOARD_MODIFIER_CTRL GLUT_ACTIVE_CTRL
	#define KEYBOARD_MODIFIER_ALT GLUT_ACTIVE_ALT

	#define MOUSE_BUTTON_DOWN GLUT_DOWN
	#define MOUSE_BUTTON_UP GLUT_UP

	#define KEYBOARD_KEYCODE_TAB 9
	#define KEYBOARD_KEYCODE_TAB_SHIFT 25
	#define KEYBOARD_KEYCODE_BACKSPACE 8
	#define KEYBOARD_KEYCODE_RETURN 13
	#define KEYBOARD_KEYCODE_DELETE 46
	#define KEYBOARD_KEYCODE_SPACE 32
	#define KEYBOARD_KEYCODE_LEFT GLUT_KEY_LEFT
	#define KEYBOARD_KEYCODE_UP GLUT_KEY_UP
	#define KEYBOARD_KEYCODE_RIGHT GLUT_KEY_RIGHT
	#define KEYBOARD_KEYCODE_DOWN GLUT_KEY_DOWN
	#define KEYBOARD_KEYCODE_POS1 106
	#define KEYBOARD_KEYCODE_END 107
	#define KEYBOARD_KEYCODE_ESCAPE 27
#endif

#include <tdme/tdme.h>
#include <tdme/application/fwd-tdme.h>

/** 
 * Application input event handler interface
 * @author Andreas Drewke
 * @version $Id$
 */
class tdme::application::InputEventHandler
{
public:
	/**
	 * Destructor
	 */
	virtual ~InputEventHandler();

	/**
	 * Get keyboard modifiers
	 * @return modifiers (one of KEYBOARD_MODIFIER_*)
	 */
	static int getKeyboardModifiers();

	/**
	 * On key down
	 * @param key key
	 * @param x x
	 * @param y y
	 */
	virtual void onKeyDown (unsigned char key, int x, int y) = 0;

	/**
	 * On key up
	 * @param key key
	 * @param x x
	 * @param y y
	 */
	virtual void onKeyUp(unsigned char key, int x, int y) = 0;

	/**
	 * On special key up
	 * @param key key
	 * @param x x
	 * @param y y
	 */
	virtual void onSpecialKeyDown (int key, int x, int y) = 0;

	/**
	 * On special key up
	 * @param key key
	 * @param x x
	 * @param y y
	 */
	virtual void onSpecialKeyUp(int key, int x, int y) = 0;

	/**
	 * On mouse dragged
	 * @param x x
	 * @param y y
	 */
	virtual void onMouseDragged(int x, int y) = 0;

	/**
	 * On mouse moved
	 * @param x x
	 * @param y y
	 */
	virtual void onMouseMoved(int x, int y) = 0;

	/**
	 * On mouse moved
	 * @param button button
	 * @param state state
	 * @param x x
	 * @param y y
	 */
	virtual void onMouseButton(int button, int state, int x, int y) = 0;

	/**
	 * On mouse wheen
	 * @param button button
	 * @param direction direction
	 * @param x x
	 * @param y y
	 */
	virtual void onMouseWheel(int button, int direction, int x, int y) = 0;

};
