#ifndef _MV_STREAM_H_
#define _MV_STREAM_H_

#include <ESPAsyncWebServer.h>

class BufferedStream : public AsyncAbstractResponse {
    private:
        cbuf *_content;
    
    public:
        BufferedStream() {
            _content = new cbuf(1024);
        }

        virtual ~BufferedStream() {
            delete _content;
        }

        bool _sourceValid() const {
            return true;
        }

        size_t _fillBuffer(uint8_t *buf, size_t maxLen) override {
            if (_content->empty()) {
                run();

                if (_content->empty()) {
                    return RESPONSE_TRY_AGAIN;
                }
            }
            
            return _content->read((char*)buf, maxLen);
        }

    protected:
        virtual void run() = 0;
        
        void send(const void *data, size_t len) {
            size_t needed = len - _content->room();
            if (needed > 0) {
                _content->resizeAdd(needed);
            }

            _content->write((const char*)data, len);
        }

        void send(const char *data) {
            send(data, strlen(data));
        }

        void send(const String &data) {
            send(data.c_str(), data.length());
        }
};

#endif