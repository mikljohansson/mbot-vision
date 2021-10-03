#include <Arduino.h>
#include "camera.h"
#include "common.h"

FrameBufferQueue *fbqueue;
TaskHandle_t cameraCaptureFrameTask;

// Framerate calculation
double windowTime = 0.0;
unsigned long windowFrames = 0;
unsigned long lastUpdatedWindow;
unsigned long lastShowedFramerate = 0;

void cameraCaptureFrame(void *p) {
    // Initialize camera
    Serial.println("Init camera");
    while (true) {
        esp_err_t err = esp_camera_init(&mv_camera_aithinker_config);
        if (err == ESP_OK) {
            break;
        }

        serialPrint("Camera probe failed with error 0x%x", err);
        delay(250);
    }

    lastUpdatedWindow = millis();

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();

        if (fb) {
            fbqueue->push(fb);

            // Calculate the framerate
            while (windowFrames >= 10.0) {
                windowTime -= (windowTime / (double)windowFrames);
                windowFrames--;
            }

            unsigned long ts = millis();
            windowTime += (ts - lastUpdatedWindow);
            windowFrames++;
            lastUpdatedWindow = ts;

            // Display the framerate
            if (ts - lastShowedFramerate > 5000) {
                serialPrint("Framerate: %02f\n", (double)windowFrames / (windowTime / 1000.0));
                lastShowedFramerate = ts;
            }
        }

        yield();
    }

    vTaskDelete(NULL);
}

void cameraRun() {
    fbqueue = new FrameBufferQueue();
    xTaskCreatePinnedToCore(cameraCaptureFrame, "captureFrame", 10000, NULL, 1, &cameraCaptureFrameTask, 0);
}

// https://github.com/geeksville/Micro-RTSP/blob/master/src/OV2640.cpp
camera_config_t mv_camera_aithinker_config {
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
    .ledc_timer = LEDC_TIMER_1,
    .ledc_channel = LEDC_CHANNEL_1,
    .pixel_format = PIXFORMAT_JPEG,
    // .frame_size = FRAMESIZE_UXGA, // needs 234K of framebuffer space
    // .frame_size = FRAMESIZE_SXGA, // needs 160K for framebuffer
    // .frame_size = FRAMESIZE_XGA, // needs 96K or even smaller FRAMESIZE_SVGA - can work if using only 1 fb
    //.frame_size = FRAMESIZE_QVGA,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 25, //0-63 lower numbers are higher quality
    .fb_count = 3       // if more than one i2s runs in continous mode.  Use only with jpeg
};
