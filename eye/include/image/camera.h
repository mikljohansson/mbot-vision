#ifndef _MV_CAMERA_H_
#define _MV_CAMERA_H_

#include <Arduino.h>
#include <esp_camera.h>
#include <vector>

class DataLogger;

class Camera {
    private:
        DataLogger &_logger;

    public:
        Camera(DataLogger &logger);
        void begin();

    private:
        void run();
        static void runStatic(void *);
};

class FrameBufferItem {
    public:
        camera_fb_t *frame;
        int readers;
        uint64_t generation;

        FrameBufferItem(camera_fb_t *fb, uint64_t generation)
         : frame(fb), readers(0), generation(generation) {}

        FrameBufferItem()
         : FrameBufferItem(0, 0) {}

        operator bool() const {
            return frame;
        }
};

class FrameBufferQueue {
    private:
        typedef std::vector<FrameBufferItem> Queue;
        Queue queue;
        SemaphoreHandle_t lock;

    public:
        FrameBufferQueue();
        ~FrameBufferQueue();

        void push(FrameBufferItem fb);
        FrameBufferItem take(FrameBufferItem last);
        FrameBufferItem poll(FrameBufferItem last);
        void release(FrameBufferItem fb);

    private:
        void expire();
};

extern FrameBufferQueue *fbqueue;

#endif
