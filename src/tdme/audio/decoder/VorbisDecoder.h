#pragma once

#include <vorbis/vorbisfile.h>

#include <string>

#include <fwd-tdme.h>
#include <tdme/audio/decoder/fwd-tdme.h>
#include <tdme/audio/decoder/AudioDecoder.h>
#include <tdme/audio/decoder/AudioDecoderException.h>
#include <tdme/os/fwd-tdme.h>
#include <tdme/os/_FileSystemException.h>
#include <tdme/utils/fwd-tdme.h>

using std::wstring;

using tdme::audio::decoder::AudioDecoder;
using tdme::audio::decoder::AudioDecoderException;
using tdme::os::_FileSystemException;
using tdme::utils::ByteBuffer;

class tdme::audio::decoder::VorbisDecoder: public AudioDecoder
{
public:
	static constexpr int32_t CHANNELS_NONE { -1 };
	static constexpr int32_t SAMPLERATE_NONE { -1 };
	static constexpr int32_t BITSPERSAMPLES_NONE { -1 };

public:

	/**
	 * Open a local file
	 * @param path name
	 * @param file name
	 */
	virtual void openFile(const wstring& pathName, const wstring& fileName) throw (_FileSystemException, AudioDecoderException);

	/**
	 * Resets this audio decoder, if a stream was open it will be rewinded
	 */
	virtual void reset() throw (_FileSystemException, AudioDecoderException);

	/**
	 * Read raw PCM data from stream
	 * @param byte buffer
	 * @return number of bytes read
	 */
	virtual int32_t readFromStream(ByteBuffer* data) throw (_FileSystemException, AudioDecoderException);

	/**
	 * Closes the audio file
	 */
	virtual void close() throw (_FileSystemException, AudioDecoderException);

	/**
	 * Constructor
	 */
	VorbisDecoder();

private:
	wstring pathName;
	wstring fileName;
	OggVorbis_File vf;
	int section;
};
