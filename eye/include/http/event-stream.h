
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include <WebServer.h>
#include "framerate.h"
#include "detection/detector.h"

class EventStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;
        Detector &_detector;

    public:
        EventStream(const WiFiClient &client, Detector &detector)
         : _client(client), _framerate("Event stream framerate: %02f\n"), _detector(detector) {}

        void start() {
            Serial.print("Event stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "eventStream", 10000, this, 3, &_task, 0);
        }

    private:
        void send(const void *data, size_t len) {
            long started = millis();
            size_t written = 0;

            while (data && _client.connected() && written < len) {
                written += _client.write(((const uint8_t *)data) + written, len - written);
                
                /*if (millis() - started > 1000) {
                    Serial.print("Event stream client timeout: ");
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
            Serial.println("Starting object detection event stream");
            
            send("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            _framerate.init();

            while (true) {
                DetectedObject blob = _detector.wait();
                String data = "data: ";
                blob.serialize(data);
                data += "\r\n\r\n";
                send(data.c_str(), data.length());

                _framerate.tick();

                if (!_client.connected()) {
                    Serial.println("Object detection event stream disconnected");
                    break;
                }

                // Let other tasks run too
                backoff();
            }
        }

        static void runStatic(void *p) {
            EventStream *stream = (EventStream *)p;
            stream->run();
            delete stream;
            vTaskDelete(NULL);
        }
};

#endif