
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
         : _client(client), _framerate("Stream framerate: %02f\n"), _showDetector(showDetector), _detector(detector) {}

        void start() {
            Serial.print("Stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "jpegStream", 10000, this, 2, &_task, 1);
        }

    private:
        void send(const void *data, size_t len) {
            for (size_t written = 0; data && _client.connected() && written < len; yield()) {
                written += _client.write(((const uint8_t *)data) + written, len - written);
            }
        }

        void send(const char *data) {
            send(data, strlen(data));
        }

        void run() {
            Serial.println("Starting mjpeg stream");
            JpegDecoder *decoder = 0;

            send("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p\r\n");
            _framerate.init();

            while (true) {
                camera_fb_t *fb = fbqueue->take();

                if (fb) {
                    String headers = "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\nContent-Type: image/jpeg\r\n";

                    if (fb->format == PIXFORMAT_JPEG && !_showDetector) {
                        headers += "Content-Length: ";
                        headers += fb->len;
                        headers += "\r\n";
                    }

                    headers += "\r\n";
                    send(headers.c_str(), headers.length());

                    if (fb->format == PIXFORMAT_JPEG) {
                        if (_showDetector) {
                            if (!decoder) {
                                decoder = new JpegDecoder();
                            }

                            if (decoder->decompress(fb->buf, fb->len)) {
                                size_t framelen = decoder->getOutputWidth() * decoder->getOutputHeight() * 3;
                                //_detector.debug(decoder->getOutputFrame(), decoder->getOutputWidth(), decoder->getOutputHeight());
                                
                                // Encode to JPEG and send
                                if (!fmt2jpg_cb(
                                        decoder->getOutputFrame(), framelen, decoder->getOutputWidth(), decoder->getOutputHeight(), 
                                        PIXFORMAT_RGB888, 40, sendStatic, this)) {
                                    Serial.println("Failed to convert framebuffer to jpeg");
                                }
                            }
                        }
                        else {
                            send(fb->buf, fb->len);
                        }
                    }
                    else if (!frame2jpg_cb(fb, 40, sendStatic, this)) {
                        Serial.println("Failed to convert framebuffer to jpeg");
                    }

                    fbqueue->release(fb);
                    _framerate.tick();
                }

                if (!_client.connected()) {
                    Serial.print("HTTP stream disconnected");
                    break;
                }
            }

            delete decoder;
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