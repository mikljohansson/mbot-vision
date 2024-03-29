#ifndef _MV_JPEG_H_
#define _MV_JPEG_H_

#include <Arduino.h>
#include <tjpgd.h>

class JpegDecoder {
    private:
        JDEC _jdec;
        uint8_t _work[TJPGD_WORKSPACE_SIZE];
        uint8_t _scaleFactor;
        
        uint8_t *_output;
        size_t _outputwidth, _outputheight;

        const uint8_t *_input;
        size_t _inputlen, _inputoffset;

    public:
        JpegDecoder(uint8_t scaleFactor);
        ~JpegDecoder();

        uint8_t *getOutputFrame();
        size_t getOutputWidth();
        size_t getOutputHeight();

        bool decompress(const uint8_t *input, size_t len);

    private:
        size_t read(JDEC *jd, uint8_t *buff, size_t nbyte);
        static size_t readStatic(JDEC *jd, uint8_t *buff, size_t nbyte);

        int write(JDEC *jd, void *bitmap, JRECT *rect);
        static int writeStatic(JDEC *jd, void *bitmap, JRECT *rect);
};

#endif
