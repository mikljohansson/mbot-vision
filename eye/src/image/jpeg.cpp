#include "mbot-vision/image/jpeg.h"

// http://elm-chan.org/fsw/tjpgd/en/appnote.html
#include <tjpgdcnf.h>
#include <tjpgd.h>

// Downscale 1/4 (1 << 2)
#define JPEG_SCALE_FACTOR   3

JpegDecoder::JpegDecoder() {
    _output = 0;
}

JpegDecoder::~JpegDecoder() {
    free(_output);
}

uint8_t *JpegDecoder::getOutputFrame() {
    return _output;
}

size_t JpegDecoder::getOutputWidth() {
    return _outputwidth;
}

size_t JpegDecoder::getOutputHeight() {
    return _outputheight;
}

bool JpegDecoder::decompress(const uint8_t *input, size_t len) {
    _input = input;
    _inputlen = len;
    _inputoffset = 0;
    
    JRESULT res = jd_prepare(&_jdec, readStatic, _work, TJPGD_WORKSPACE_SIZE, this);
    
    if (res != JDR_OK) {
        Serial.println("Failed to prepare JPEG decoder");
        return false;
    }

    // Allocate output frame buffer if needed
    if (!_output) {
        _outputwidth = (_jdec.width >> JPEG_SCALE_FACTOR);
        _outputheight = (_jdec.height >> JPEG_SCALE_FACTOR);
        _output = (uint8_t*)malloc(JD_BPP * _outputwidth * _outputheight);
    }

    res = jd_decomp(&_jdec, writeStatic, JPEG_SCALE_FACTOR);
    
    if (res != JDR_OK) {
        Serial.println("Failed to decode JPEG image");
        return false;
    }

    return true;
}

size_t JpegDecoder::read(JDEC *jd, uint8_t *buff, size_t nbyte) {
    nbyte = min(nbyte, _inputlen - _inputoffset);

    if (buff) {
        memcpy(buff, _input + _inputoffset, nbyte);
    } 
    
    _inputoffset += nbyte;

    yield();
    return nbyte;
}

size_t JpegDecoder::readStatic(JDEC *jd, uint8_t *buff, size_t nbyte) {
    JpegDecoder *decoder = (JpegDecoder *)jd->device;
    return decoder->read(jd, buff, nbyte);
}

int JpegDecoder::write(JDEC *jd, void *bitmap, JRECT *rect) {
    const uint8_t *source = (const uint8_t *)bitmap;
    uint8_t *target = _output + (rect->top * _outputwidth + rect->left) * JD_BPP;
    
    for (size_t y = rect->top; y <= rect->bottom; y++) {
        for (size_t x = rect->left; x <= rect->right; x++) {
            target[0] = source[2];  // Swap red and blue which are getting mixed up for some reason
            target[1] = source[1];
            target[2] = source[0];

            target += JD_BPP;
            source += JD_BPP;
        }

        target += (_outputwidth - rect->right - 1 + rect->left) * JD_BPP;
    }

    yield();
    return 1;    /* Continue to decompress */
}

int JpegDecoder::writeStatic(JDEC *jd, void *bitmap, JRECT *rect) {
    JpegDecoder *decoder = (JpegDecoder *)jd->device;
    return decoder->write(jd, bitmap, rect);
}
