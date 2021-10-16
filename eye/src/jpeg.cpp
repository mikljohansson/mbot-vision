// http://elm-chan.org/fsw/tjpgd/en/appnote.html
#include <tjpgdcnf.h>
#include <tjpgd.h>
#include "jpeg.h"

/* Bytes per pixel of image output */
#define N_BPP               (3 - JD_FORMAT)
#define JPEG_WORK_SIZE      (3500 + 320 + (6 << 10))

// Downscale 1/8 (1 << 3)
#define JPEG_SCALE_FACTOR   3

JpegDecoder::JpegDecoder() {
    _jdec = new JDEC();
    _work = malloc(JPEG_WORK_SIZE);
    _output = 0;
}

JpegDecoder::~JpegDecoder() {
    delete _jdec;
    free(_work);
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
    
    JRESULT res = jd_prepare(_jdec, readStatic, _work, JPEG_WORK_SIZE, this);
    
    if (res != JDR_OK) {
        Serial.println("Failed to prepare JPEG decoder");
        return false;
    }

    // Allocate output frame buffer if needed
    if (!_output) {
        _outputwidth = (_jdec->width >> JPEG_SCALE_FACTOR);
        _outputheight = (_jdec->height >> JPEG_SCALE_FACTOR);
        _output = (uint8_t*)malloc(N_BPP * _outputwidth * _outputheight);
    }

    res = jd_decomp(_jdec, writeStatic, JPEG_SCALE_FACTOR);
    
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
    return nbyte;
}

size_t JpegDecoder::readStatic(JDEC *jd, uint8_t *buff, size_t nbyte) {
    JpegDecoder *decoder = (JpegDecoder *)jd->device;
    return decoder->read(jd, buff, nbyte);
}

int JpegDecoder::write(JDEC *jd, void *bitmap, JRECT *rect) {
    uint8_t *src, *dst;
    uint16_t y, bws;
    unsigned int bwd;

    /* Copy the output image rectanglar to the frame buffer */
    src = (uint8_t*)bitmap;
    dst = _output + N_BPP * (rect->top * _outputwidth + rect->left);  /* Left-top of destination rectangular */
    bws = N_BPP * (rect->right - rect->left + 1);     /* Width of output rectangular [byte] */
    bwd = N_BPP * _outputwidth;                         /* Width of frame buffer [byte] */
    
    for (y = rect->top; y <= rect->bottom; y++) {
        memcpy(dst, src, bws);   /* Copy a line */
        src += bws; dst += bwd;  /* Next line */
    }

    return 1;    /* Continue to decompress */
}

int JpegDecoder::writeStatic(JDEC *jd, void *bitmap, JRECT *rect) {
    JpegDecoder *decoder = (JpegDecoder *)jd->device;
    return decoder->write(jd, bitmap, rect);
}
