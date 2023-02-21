
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include <ESPAsyncWebServer.h>
#include "buffered-stream.h"
#include "framerate.h"
#include "detection/detector.h"

class EventStream : public BufferedStream {
    private:
        Framerate _framerate;
        Detector &_detector;
        long _last;

    public:
        EventStream(Detector &detector)
         : _framerate("Event stream framerate: %02f\n"), _detector(detector), _last(0) {
            _code = 200;
            _contentType = "text/event-stream";
            _sendContentLength = false;

            _framerate.init();
        }

        virtual ~EventStream() {
            Serial.println("Event stream disconnected");
        }

    private:
        void run() {
            long ts = millis();
            if (ts - _last > 100) {
                _last = ts;

                DetectedObject blob = _detector.get();
                String data = "data: ";
                blob.serialize(data);
                data += "\r\n\r\n";
                send(data.c_str(), data.length());

                _framerate.tick();
            }
        }
};

#endif