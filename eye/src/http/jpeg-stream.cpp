#include "mbot-vision/http/jpeg-stream.h"

#include <memory>
#include "mbot-vision-model.h"
#include "mbot-vision-config.h"

static volatile bool showDetectorStream = false;

JpegStream::JpegStream(httpd_req_t *request, Detector &detector)
 : AsyncStream(request), 
   _framerate("MJPEG stream framerate: %02f\n"), _detector(detector), _disconnected(false) {}

void JpegStream::toggleDetectorStream() {
    showDetectorStream = !showDetectorStream;
}

void JpegStream::run() {
    Serial.println("Starting MJPEG stream");
    std::unique_ptr<JpegDecoder> decoder;

    if (!send("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p\r\n")) {
        return;
    }

    _framerate.init();
    FrameBufferItem fb;

    while (true) {
        fb = fbqueue->take(fb);

        String headers = "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n";

        if (!showDetectorStream) {
            headers += "Content-Type: image/jpeg\r\n";
        }
        else {
            headers += "Content-Type: image/bmp\r\n";
        }

        if (fb.frame->format == PIXFORMAT_JPEG && !showDetectorStream) {
            headers += "Content-Length: ";
            headers += fb.frame->len;
            headers += "\r\n";
        }

        headers += "\r\n";
        
        if (!send(headers.c_str(), headers.length())) {
            fbqueue->release(fb);
            return;
        }

        if (fb.frame->format == PIXFORMAT_JPEG) {
            if (showDetectorStream) {
                if (!decoder) {
                    decoder.reset(new JpegDecoder((int)sqrt(MV_CAM_WIDTH / MBOT_VISION_MODEL_INPUT_WIDTH)));
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
                if (!send(fb.frame->buf, fb.frame->len)) {
                    fbqueue->release(fb);
                    return;
                }
            }
        }
        else if (!frame2jpg_cb(fb.frame, 60, sendStatic, this) || _disconnected) {
            Serial.println("Failed to convert framebuffer to jpeg");
            fbqueue->release(fb);
            return;
        }

        fbqueue->release(fb);
        _framerate.tick();

        // Let other tasks run too
        delay(50);
    }
}

size_t JpegStream::sendStatic(void *p, size_t index, const void *data, size_t len) {
    JpegStream *stream = (JpegStream *)p;
    
    if (!stream->send(data, len)) {
        stream->_disconnected = true;
    }

    return len;
}
