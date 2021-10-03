#ifndef _MV_JPEG_H_
#define _MV_JPEG_H_

#include <Arduino.h>
#include <tjpgd.h>

#define JPEG_WORK_SIZE 3500

// http://elm-chan.org/fsw/tjpgd/en/appnote.html
class JpegDecoder {
    private:
        JDEC _jdec;
        void *_work;

    public:
        JpegDecoder()
         : _work(malloc(JPEG_WORK_SIZE)) {}

        ~JpegDecoder() {
            free(_work);
        }
        
        void decompress(const uint8_t *buf, size_t len, uint8_t *output) {
            JRESULT res = jd_prepare(&jdec, in_func, work, sz_work, &devid);

        }
};

#endif