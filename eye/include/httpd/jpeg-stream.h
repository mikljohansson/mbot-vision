
#ifndef _MV_JPEG_STREAM_H_
#define _MV_JPEG_STREAM_H_

#include <Arduino.h>
#include <WebServer.h>
#include "camera.h"
#include "framerate.h"
#include "jpeg.h"

class JpegStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;
        bool _showDetector;

    public:
        JpegStream(const WiFiClient &client, bool showDetector)
         : _client(client), _framerate("Stream framerate: %02f\n"), _showDetector(showDetector) {}

        void start() {
            Serial.print("Stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "jpegStream", 10000, this, 3, &_task, 1);
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
                            JpegDecoder decoder;
                            decoder.decompress(fb->buf, fb->len);
                            size_t framelen = decoder.getOutputWidth() * decoder.getOutputHeight() * 3;
                            
                            // Convert to HSV and back to RGB
                            for (uint8_t *pixel = decoder.getOutputFrame(), *last = pixel + framelen; pixel < last; pixel += 3) {
                                RgbColor rgb = HsvToRgb(RgbToHsv({pixel[0], pixel[1], pixel[2]}));
                                pixel[0] = rgb.r;
                                pixel[1] = rgb.g;
                                pixel[2] = rgb.b;
                            }

                            // Encode to JPEG and send
                            if (!fmt2jpg_cb(
                                    decoder.getOutputFrame(), framelen, decoder.getOutputWidth(), decoder.getOutputHeight(), 
                                    PIXFORMAT_RGB888, 40, sendStatic, this)) {
                                Serial.println("Failed to convert framebuffer to jpeg");
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
                    Serial.print("HTTP stream disconnected: ");
                    Serial.println(_client.remoteIP());
                    break;
                }
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