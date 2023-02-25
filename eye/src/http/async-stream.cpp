#include "mbot-vision/http/async-stream.h"

#include <Arduino.h>

AsyncStream::AsyncStream(httpd_req_t *request) {
    _handle = request->handle;
    _sockfd = httpd_req_to_sockfd(request);
    _backoff = 1;
}

AsyncStream::~AsyncStream() {}

bool AsyncStream::send(const void *data, size_t len) {
    size_t written = 0;

    while (data && written < len) {
        int res = httpd_socket_send(_handle, _sockfd, ((const char *)data) + written, std::min(len - written, 1024U), 0);

        if (res < 0) {
            Serial.println(String("Async stream disconnected with error ") + res);
            httpd_sess_trigger_close(_handle, _sockfd);
            return false;
        }

        written += res;

        // Prevents watchdog from triggering and use some dynamic backoff flow control
        _backoff = res > 0 ? std::max(_backoff / 2, 1L) : std::min((long)(_backoff * 1.33 + 1), 100L);
        delay(_backoff);
    }

    return true;
}

bool AsyncStream::send(const char *data) {
    return send(data, strlen(data));
}
