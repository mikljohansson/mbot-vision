
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include <WebServer.h>
#include "framerate.h"
#include "blobdetector.h"

class EventStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;
        BlobDetector &_detector;

    public:
        EventStream(const WiFiClient &client, BlobDetector &detector)
         : _client(client), _framerate("Event stream framerate: %02f\n"), _detector(detector) {}

        void start() {
            Serial.print("Event stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "eventStream", 10000, this, 3, &_task, 1);
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
            Serial.println("Starting event stream");
            
            send("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            _framerate.init();

            while (true) {
                DetectedBlob blob = _detector.wait();
                
                String data = "data: ";
                blob.serialize(data);
                data += "\r\n\r\n";
                send(data.c_str(), data.length());

                _framerate.tick();

                if (!_client.connected()) {
                    Serial.print("HTTP event stream disconnected");
                    break;
                }
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