
#ifndef _MV_JPEG_STREAM_H_
#define _MV_JPEG_STREAM_H_

#include <Arduino.h>
#include <WebServer.h>
#include "framerate.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "detection/detector.h"

class JpegStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;
        bool _showDetector;
        Detector &_detector;

    public:
        JpegStream(const WiFiClient &client, bool showDetector, Detector &detector)
         : _client(client), _framerate("MJPEG stream framerate: %02f\n"), _showDetector(showDetector), _detector(detector) {}

        void start() {
            Serial.print("Stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "jpegStream", 10000, this, 2, &_task, 0);
        }

    private:
        void send(const void *data, size_t len) {
            long started = millis();
            size_t written = 0;

            while (data && _client.connected() && written < len) {
                written += _client.write(((const uint8_t *)data) + written, len - written);

                /*if (millis() - started > 1000) {
                    Serial.print("MJPEG stream client timeout: ");
                    Serial.println(_client.remoteIP());
                    close(_client.fd());
                    _client.stop();
                    return;
                }*/

                // Prevents watchdog from triggering
                delay(1);
            }
        }

        void send(const char *data) {
            send(data, strlen(data));
        }

        void run() {
            Serial.println("Starting MJPEG stream");
            std::unique_ptr<JpegDecoder> decoder;

            send("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p\r\n");
            _framerate.init();
            FrameBufferItem fb;

            while (true) {
                fb = fbqueue->take(fb);

                String headers = "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n";

                if (!_showDetector) {
                    headers += "Content-Type: image/jpeg\r\n";
                }
                else {
                    headers += "Content-Type: image/bmp\r\n";
                }

                if (fb.frame->format == PIXFORMAT_JPEG && !_showDetector) {
                    headers += "Content-Length: ";
                    headers += fb.frame->len;
                    headers += "\r\n";
                }

                headers += "\r\n";
                send(headers.c_str(), headers.length());

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

                if (!_client.connected()) {
                    Serial.println("MJPEG stream disconnected");
                    break;
                }

                // Let other tasks run too
                delay(1);
            }
        }

        static void runStatic(void *p) {
            JpegStream *stream = (JpegStream *)p;
            stream->run();
            delete stream;
            vTaskDelete(NULL);
        }

        static size_t sendStatic(void *p, size_t index, const void *data, size_t len) {
            JpegStream *stream = (JpegStream *)p;
            stream->send(data, len);
            return len;
        }
};

#endif