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
        camera_fb_t *fb;
        int readers;

        FrameBufferItem(camera_fb_t *fb) {
            this->fb = fb;
            this->readers = 0;
        }
};

class FrameBufferQueue {
    private:
        typedef std::vector<FrameBufferItem> Queue;
        Queue queue;
        SemaphoreHandle_t lock;
        SemaphoreHandle_t signal;

    public:
        FrameBufferQueue();
        ~FrameBufferQueue();

        void push(camera_fb_t *fb);
        camera_fb_t *take();
        void release(camera_fb_t *fb);

    private:
        void expire();
};

extern FrameBufferQueue *fbqueue;

#endif
