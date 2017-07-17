// Generated from /tdme/src/tdme/utils/ArrayListIteratorMultiple.java

#pragma once

#include <vector>
#include <algorithm>

#include <fwd-tdme.h>
#include <java/lang/fwd-tdme.h>
#include <java/util/fwd-tdme.h>
#include <tdme/utils/fwd-tdme.h>
#include <java/lang/Object.h>
#include <java/util/Iterator.h>
#include <java/lang/Iterable.h>

using std::vector;
using std::find;

using java::lang::Object;
using java::util::Iterator;
using java::lang::Iterable;

struct default_init_tag;

namespace tdme {
namespace utils {

/** 
 * Array list iterator for multiple array lists
 * @author Andreas Drewke
 * @version $Id$
 */
template<typename T>
class ArrayListIteratorMultiple final
	: public Iterator
	, public Iterable
{

public:
	typedef Object super;

private:
	int32_t vectorIdx {  0 };
	int32_t elementIdx {  0 };
	int32_t length { 0  };
	vector<vector<T>*> arrayLists {  };
public:

	/** 
	 * Clears list of array lists to iterate
	 */
	void clear() {
		arrayLists.clear();
	}

	/** 
	 * Adds array lists to iterate
	 * @param array list
	 */
	void addArrayList(vector<T>* _arrayList) {
		if (find(arrayLists.begin(), arrayLists.end(), _arrayList) != arrayLists.end()) return;
		arrayLists.push_back(_arrayList);
	}

	/** 
	 * resets vector iterator for iterating
	 * @return this vector iterator
	 */
	ArrayListIteratorMultiple<T>* reset() {
		this->vectorIdx = 0;
		this->elementIdx = 0;
		this->length = 0;
		for (auto i = 0; i < arrayLists.size(); i++) {
			this->length += arrayLists.at(i)->size();
		}
		return this;
	}

	bool hasNext() override {
		auto hasNext = (vectorIdx < arrayLists.size() - 1) || (vectorIdx == arrayLists.size() - 1 && elementIdx < arrayLists.at(vectorIdx)->size());
		return hasNext;
	}

	T next() override {
		auto element = arrayLists.at(vectorIdx)->at(elementIdx++);
		if (elementIdx == arrayLists.at(vectorIdx)->size()) {
			elementIdx = 0;
			vectorIdx++;
		}
		return element;
	}

	void remove() override {

	}

	Iterator* iterator() {
		reset();
		return this;
	}

	/** 
	 * Clones this iterator
	 */
	ArrayListIteratorMultiple<T>* clone() override {
		return new ArrayListIteratorMultiple<T>(arrayLists);
	}

	// Generated
	ArrayListIteratorMultiple() {
		reset();
	}

	/**
	 * Adds array lists to iterate
	 * @param array list
	 */
	ArrayListIteratorMultiple(vector<vector<T>*>& arrayLists) {
		this->arrayLists = arrayLists;
		reset();
	}

};

};
};
