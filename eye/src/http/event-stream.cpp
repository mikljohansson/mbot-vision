#include "mbot-vision/http/event-stream.h"

#include "mbot-vision/common.h"

EventStream::EventStream(httpd_req_t *request, Detector &detector)
 : AsyncStream(request), 
   _framerate("Event stream framerate: %02f\n"), _detector(detector) {}

void EventStream::run() {
    Serial.println("Starting object detection event stream");
    
    if (!send("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\n\r\n")) {
        return;
    }

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
