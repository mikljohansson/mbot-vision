
#ifndef _MV_JPEG_STREAM_H_
#define _MV_JPEG_STREAM_H_

#include <Arduino.h>
#include <ESPAsyncWebServer.h>
#include "buffered-stream.h"
#include "framerate.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "detection/detector.h"

class JpegStream : public BufferedStream {
    private:
        Framerate _framerate;
        bool _showDetector;
        Detector &_detector;

        std::unique_ptr<JpegDecoder> decoder;
        FrameBufferItem fb;

    public:
        JpegStream(bool showDetector, Detector &detector)
         : _framerate("MJPEG stream framerate: %02f\n"), _showDetector(showDetector), _detector(detector) {
            _code = 200;
            _contentType = "multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p";
            _sendContentLength = false;

            _framerate.init();
        }

        virtual ~JpegStream() {
            Serial.println("MJPEG stream disconnected");
        }

    private:
        void run() {
            fb = fbqueue->poll(fb);
            if (!fb) {
                return;
            }

            send("\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n");

            if (!_showDetector) {
                send("Content-Type: image/jpeg\r\n");
            }
            else {
                send("Content-Type: image/bmp\r\n");
            }

            if (fb.frame->format == PIXFORMAT_JPEG && !_showDetector) {
                send(String("Content-Length: ") + fb.frame->len + "\r\n");
            }

            send("\r\n");

            if (fb.frame->format == PIXFORMAT_JPEG) {
                if (_showDetector) {
                    if (!decoder) {
                        decoder.reset(new JpegDecoder());
                    }

                    if (decoder->decompress(fb.frame->buf, fb.frame->len)) {
                        _detector.draw(decoder->getOutputFrame(), decoder->getOutputWidth(), decoder->getOutputHeight());
                        
                        // Encode to JPEG and send
                        /*
                        if (!fmt2jpg_cb(
                                decoder->getOutputFrame(), 
                                decoder->getOutputWidth() * decoder->getOutputHeight() * JD_BPP, 
                                decoder->getOutputWidth(), decoder->getOutputHeight(), 
                                PIXFORMAT_RGB888, 80, sendStatic, this)) {
                            Serial.println("Failed to convert framebuffer to jpeg");
                        }
                        */

                        // Encode to lossless BMP and send
                        uint8_t *out_bmp = 0;
                        size_t out_bmp_len;

                        if (fmt2bmp(
                            decoder->getOutputFrame(), 
                            decoder->getOutputWidth() * decoder->getOutputHeight() * JD_BPP,
                            decoder->getOutputWidth(), decoder->getOutputHeight(), 
                            PIXFORMAT_RGB888, &out_bmp, &out_bmp_len)) {
                            send(out_bmp, out_bmp_len);
                        }
                        else {
                            Serial.println("Failed to convert framebuffer to jpeg");
                        }
                        
                        free(out_bmp);
                    }
                }
                else {
                    send(fb.frame->buf, fb.frame->len);
                }
            }
            else if (!frame2jpg_cb(fb.frame, 60, sendStatic, this)) {
                Serial.println("Failed to convert framebuffer to jpeg");
            }

            fbqueue->release(fb);
            _framerate.tick();
        }

        static size_t sendStatic(void *p, size_t index, const void *data, size_t len) {
            JpegStream *stream = (JpegStream *)p;
            stream->send(data, len);
            return len;
        }
};

#endif