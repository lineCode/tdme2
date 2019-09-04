#pragma once

#include <vorbis/vorbisfile.h>

#include <string>
#include <vector>

#include <tdme/tdme.h>
#include <tdme/audio/decoder/fwd-tdme.h>
#include <tdme/audio/decoder/AudioDecoder.h>
#include <tdme/audio/decoder/AudioDecoderException.h>
#include <tdme/os/filesystem/fwd-tdme.h>
#include <tdme/os/filesystem/FileSystemException.h>
#include <tdme/utils/fwd-tdme.h>

using std::string;
using std::vector;

using tdme::audio::decoder::AudioDecoder;
using tdme::audio::decoder::AudioDecoderException;
using tdme::os::filesystem::FileSystemException;
using tdme::utils::ByteBuffer;

/**
 * OGG/Vorbis audio decoder
 * @author Andreas Drewke
 */
class tdme::audio::decoder::VorbisDecoder: public AudioDecoder
{
public:
	struct OGGFileData {
		vector<uint8_t> data;
		size_t position { 0 };
	};

	/**
	 * Open a local file
	 * @param pathName path name
	 * @param fileName file name
	 * @throws tdme::os::filesystem::FileSystemException
	 * @throws tdme::audio::decoder::AudioDecoderException
	 */
	virtual void openFile(const string& pathName, const string& fileName);

	/**
	 * Resets this audio decoder, if a stream was open it will be rewinded
	 * @throws tdme::os::filesystem::FileSystemException
	 * @throws tdme::audio::decoder::AudioDecoderException
	 */
	virtual void reset();

	/**
	 * Read raw PCM data from stream
	 * @param data byte buffer
	 * @throws tdme::os::filesystem::FileSystemException
	 * @throws tdme::audio::decoder::AudioDecoderException
	 * @return number of bytes read
	 */
	virtual int32_t readFromStream(ByteBuffer* data);

	/**
	 * Closes the audio file
	 * @throws tdme::os::filesystem::FileSystemException
	 * @throws tdme::audio::decoder::AudioDecoderException
	 */
	virtual void close();

	/**
	 * Constructor
	 */
	VorbisDecoder();

private:
	OGGFileData* oggFileData { nullptr };
	string pathName;
	string fileName;
	OggVorbis_File vf;
	int section;
};
