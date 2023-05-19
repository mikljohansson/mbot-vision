#include "mbot-vision/image/camera.h"

#include <Arduino.h>
#include <esp_camera.h>
#include "mbot-vision/framerate.h"
#include "mbot-vision/common.h"
#include "mbot-vision/wiring.h"
#include "mbot-vision/datalog.h"

FrameBufferQueue *fbqueue;
static TaskHandle_t cameraTask;
static Framerate framerate("Camera framerate: %02f\n");

#ifdef BOARD_ESP32S3CAM

// https://github.com/Freenove/Freenove_ESP32_S3_WROOM_Board
camera_config_t mv_camera_config {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = 15,
    .pin_sscb_sda = 4,
    .pin_sscb_scl = 5,
    .pin_d7 = 16,
    .pin_d6 = 17,
    .pin_d5 = 18,
    .pin_d4 = 12,
    .pin_d3 = 10,
    .pin_d2 = 8,
    .pin_d1 = 9,
    .pin_d0 = 11,
    .pin_vsync = 6,
    .pin_href = 7,
    .pin_pclk = 13,
    
    .xclk_freq_hz = 20000000,
    .ledc_timer = MV_CAM_TIMER,
    .ledc_channel = MV_CAM_CHAN,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
    // .frame_size = FRAMESIZE_XGA, // needs 96K or even smaller FRAMESIZE_SVGA - can work if using only 1 fb
    //.frame_size = FRAMESIZE_QVGA,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 40, //0-63 lower numbers are higher quality
    .fb_count = 2       // if more than one i2s runs in continous mode.  Use only with jpeg
};


#else

// https://github.com/geeksville/Micro-RTSP/blob/master/src/OV2640.cpp
// https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/
camera_config_t mv_camera_config {
    .pin_pwdn = 32,
    .pin_reset = -1,

    .pin_xclk = 0,

    .pin_sscb_sda = 26,
    .pin_sscb_scl = 27,

    // Note: LED GPIO is apparently 4 not sure where that goes
    // per https://github.com/donny681/ESP32_CAMERA_QR/blob/e4ef44549876457cd841f33a0892c82a71f35358/main/led.c
    .pin_d7 = 35,
    .pin_d6 = 34,
    .pin_d5 = 39,
    .pin_d4 = 36,
    .pin_d3 = 21,
    .pin_d2 = 19,
    .pin_d1 = 18,
    .pin_d0 = 5,
    .pin_vsync = 25,
    .pin_href = 23,
    .pin_pclk = 22,
    .xclk_freq_hz = 20000000,
    .ledc_timer = MV_CAM_TIMER,
    .ledc_channel = MV_CAM_CHAN,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
    // .frame_size = FRAMESIZE_XGA, // needs 96K or even smaller FRAMESIZE_SVGA - can work if using only 1 fb
    //.frame_size = FRAMESIZE_QVGA,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 40, //0-63 lower numbers are higher quality
    .fb_count = 2       // if more than one i2s runs in continous mode.  Use only with jpeg
};

#endif

FrameBufferQueue::FrameBufferQueue() {
    // Create an unlocked mutex to manipulate the queue
    lock = xSemaphoreCreateMutex();
    xSemaphoreGive(lock);
}

FrameBufferQueue::~FrameBufferQueue() {
    vSemaphoreDelete(lock);
}

void FrameBufferQueue::push(FrameBufferItem fb) {
    // Add framebuffer into queue and signal consumers
    if (xSemaphoreTake(lock, portMAX_DELAY)) {
        expire();
        queue.push_back(fb);
        xSemaphoreGive(lock);
    }
    else {
        // High lock-contention, so just release the frame back to the camera
        esp_camera_fb_return(fb.frame);
        backoff();
    }
}

FrameBufferItem FrameBufferQueue::take(FrameBufferItem last) {
    FrameBufferItem result(0, last.generation);
    
    while (true) {
        // Add a read-lock on the most recent framebuffer
        if (xSemaphoreTake(lock, portMAX_DELAY)) {
            if (!queue.empty()) {
                FrameBufferItem &item = queue.at(queue.size() - 1);
                
                // Don't consume the same item again
                if (item.generation > last.generation) {
                    item.readers++;
                    result = item;
                }
            }

            xSemaphoreGive(lock);
            
            if (result) {
                break;
            }
        }

        // Throttle the consumer in waiting for a new buffer
        backoff();
    }

    return result;
}

FrameBufferItem FrameBufferQueue::poll(FrameBufferItem last) {
    FrameBufferItem result(0, last.generation);
    
    // Add a read-lock on the most recent framebuffer
    if (xSemaphoreTake(lock, portMAX_DELAY)) {
        if (!queue.empty()) {
            FrameBufferItem &item = queue.at(queue.size() - 1);
            
            // Don't consume the same item again
            if (item.generation > last.generation) {
                item.readers++;
                result = item;
            }
        }

        xSemaphoreGive(lock);
    }

    return result;
}

void FrameBufferQueue::release(FrameBufferItem fb) {
    while (true) {
        // Release the read-lock on the framebuffer
        if (xSemaphoreTake(lock, portMAX_DELAY)) {
            for (Queue::iterator it = queue.begin(); it != queue.end(); ++it) {
                if (it->frame == fb.frame) {
                    it->readers--;
                    
                    // If this was the last reader and we have newer frames, then release the framebuffer back to 
                    // the camera. Keeping at least 1 frame in the queue, otherwise thread starvation will occur
                    if (it->readers == 0 && queue.size() > 1) {
                        esp_camera_fb_return(it->frame);
                        queue.erase(it);
                    }

                    break;
                }
            }

            xSemaphoreGive(lock);
            break;
        }

        backoff();
    }
}

void FrameBufferQueue::expire() {
    // Removes any framebuffers with 0 readers
    for (Queue::iterator it = queue.begin(); it != queue.end(); ) {
        if (it->readers == 0) {
            esp_camera_fb_return(it->frame);
            queue.erase(it);
        }
        else {
            ++it;
        }
    }
}

void Camera::run() {
    // Initialize camera
    Serial.println("Initializing camera");
    while (true) {
        esp_err_t err = esp_camera_init(&mv_camera_config);
        if (err == ESP_OK) {
            break;
        }

        serialPrint("Camera probe failed with error 0x%x", err);
        Serial.print(".");
        delay(250);
    }

    Serial.println("Camera successfully initialized");
    framerate.init();
    uint64_t generation = 1;

    while (true) {
        // Grab frame from the camera and push it into the queue
        camera_fb_t *fb = esp_camera_fb_get();

        if (fb) {
            fbqueue->push(FrameBufferItem(fb, generation));
            framerate.tick();
            generation++;
            
            if (mv_camera_config.pixel_format == PIXFORMAT_JPEG) {
                _logger.logJpeg(fb->buf, fb->len);
            }

            // Let other tasks run too
            delay(50);
        }
        else {
            // Camera was out of framebuffers, so wait a bit longer
            backoff();
        }
    }
}

void Camera::runStatic(void *p) {
    Camera *camera = (Camera *)p;
    camera->run();
    vTaskDelete(NULL);
}

Camera::Camera(DataLogger &logger)
 : _logger(logger) {}

void Camera::begin() {
    fbqueue = new FrameBufferQueue();
    xTaskCreatePinnedToCore(runStatic, "camera", 10000, this, 1, &cameraTask, 0);
}
