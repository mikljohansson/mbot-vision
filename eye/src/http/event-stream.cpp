#include "mbot-vision/http/event-stream.h"

#include "mbot-vision/common.h"

EventStream::EventStream(httpd_req_t *request, Detector &detector)
 : _framerate("Event stream framerate: %02f\n"), _detector(detector) {
    _handle = request->handle;
    _sockfd = httpd_req_to_sockfd(request);
}

void EventStream::run() {
    Serial.println("Starting object detection event stream");
    
    if (!send("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")) {
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

bool EventStream::send(const void *data, size_t len) {
    size_t written = 0;
    long ts = 1;

    while (data && written < len) {
        int res = httpd_socket_send(_handle, _sockfd, ((const char *)data) + written, len - written, 0);

        if (res < 0) {
            Serial.println(String("Event stream disconnected with error ") + res);
            httpd_sess_trigger_close(_handle, _sockfd);
            return false;
        }

        written += res;

        // Prevents watchdog from triggering
        ts = res > 0 ? std::max(ts / 2, 1L) : std::min(ts * 2, 100L);
        delay(ts);
    }

    return true;
}

bool EventStream::send(const char *data) {
    return send(data, strlen(data));
}
