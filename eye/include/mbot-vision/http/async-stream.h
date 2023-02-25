#ifndef _MV_ASYNC_STREAM_H_
#define _MV_ASYNC_STREAM_H_

#include <Arduino.h>
#include <esp_http_server.h>

class AsyncStream {
    private:
        httpd_handle_t _handle;
        int _sockfd;
        long _backoff;

    public:
        AsyncStream(httpd_req_t *request);
        virtual ~AsyncStream();

    protected:
        bool send(const void *data, size_t len);
        bool send(const char *data);
};

#endif
