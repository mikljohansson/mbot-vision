
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include <WebServer.h>
#include "framerate.h"
#include "detection/detector.h"

class EventStream {
    private:
        httpd_handle_t _handle;
        int _sockfd;
        
        TaskHandle_t _task;
        Framerate _framerate;
        Detector &_detector;

    public:
        EventStream(httpd_req_t *request, Detector &detector)
         : _framerate("Event stream framerate: %02f\n"), _detector(detector) {
            _handle = request->handle;
            _sockfd = httpd_req_to_sockfd(request);
         }

        void start() {
            xTaskCreatePinnedToCore(runStatic, "eventStream", 10000, this, 1, &_task, 0);
        }

    private:
        bool send(const void *data, size_t len) {
            size_t written = 0;

            while (data && written < len) {
                int res = httpd_socket_send(_handle, _sockfd, ((const char *)data) + written, len - written, 0);

                if (res < 0) {
                    Serial.println(String("Event stream disconnected with error ") + res);
                    httpd_sess_trigger_close(_handle, _sockfd);
                    return false;
                }

                written += res;

                // Prevents watchdog from triggering
                delay(1);
            }

            return true;
        }

        bool send(const char *data) {
            return send(data, strlen(data));
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
                
                if (!send(data.c_str(), data.length())) {
                    return;
                }

                _framerate.tick();

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