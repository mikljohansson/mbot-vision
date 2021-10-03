#ifndef _MV_CAMERA_H_
#define _MV_CAMERA_H_

#include <Arduino.h>
#include <esp_camera.h>
#include <vector>

extern camera_config_t mv_camera_aithinker_config;

void cameraRun();

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
        FrameBufferQueue() {
            // Create an unlocked mutex to manipulate the queue
            lock = xSemaphoreCreateMutex();
            xSemaphoreGive(lock);

            // Create a semaphore to signal that items are available
            signal = xSemaphoreCreateBinary();
        }

        ~FrameBufferQueue() {
            vSemaphoreDelete(lock);
            vSemaphoreDelete(signal);
        }

        void push(camera_fb_t *fb) {
            if (xSemaphoreTake(lock, portTICK_PERIOD_MS * 100)) {
                expire();
                queue.push_back(fb);
                xSemaphoreGive(lock);
                xSemaphoreGive(signal);
            }
            else {
                esp_camera_fb_return(fb);
            }
        }

        camera_fb_t *take() {
            camera_fb_t *result = 0;

            if (xSemaphoreTake(signal, portTICK_PERIOD_MS * 1000) && xSemaphoreTake(lock, portMAX_DELAY)) {
                if (!queue.empty()) {
                    FrameBufferItem &item = queue.at(queue.size() - 1);
                    result = item.fb;
                    item.readers++;
                }

                xSemaphoreGive(lock);

                if (result) {
                    // Signal other tasks too
                    xSemaphoreGive(signal);
                }
            }

            return result;
        }

        void release(camera_fb_t *fb) {
            while (true) {
                if (xSemaphoreTake(lock, portMAX_DELAY)) {
                    for (Queue::iterator it = queue.begin(); it != queue.end(); ++it) {
                        if (it->fb == fb) {
                            it->readers--;
                            
                            if (it->readers == 0) {
                                esp_camera_fb_return(it->fb);
                                queue.erase(it);
                            }

                            break;
                        }
                    }

                    xSemaphoreGive(lock);
                    break;
                }
            }
        }

    private:
        void expire() {
            for (Queue::iterator it = queue.begin(); it != queue.end(); ) {
                if (it->readers == 0) {
                    esp_camera_fb_return(it->fb);
                    queue.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
};

extern FrameBufferQueue *fbqueue;

#endif
