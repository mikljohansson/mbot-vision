#include "blobdetector.h"
#include "camera.h"
#include "jpeg.h"
#include "colorspace.h"

void DetectedBlob::serialize(String &out) {
    out += "{'x':";
    out += x;
    out += ",'y':";
    out += y;
    out += ",'width':";
    out += width;
    out += ",'height':";
    out += height;
    out += "}";
}

BlobDetector::BlobDetector(RgbColor color)
 : _color(RgbToHsv(color)), _framerate("Blob detector framerate: %02f\n") {
    _signal = xSemaphoreCreateBinary();
}

BlobDetector::~BlobDetector() {
    vSemaphoreDelete(_signal);
}

void BlobDetector::start() {
    xTaskCreatePinnedToCore(runStatic, "blobDetector", 10000, this, 1, &_task, 1);
}

DetectedBlob BlobDetector::get() {
    xSemaphoreTake(_signal, portMAX_DELAY);

    DetectedBlob detected = _detected;
    return detected;
}

void BlobDetector::run() {
    JpegDecoder decoder;
    _framerate.init();

    while (true) {
        camera_fb_t *fb = fbqueue->take();

        if (fb) {
            bool result = decoder.decompress(fb->buf, fb->len);
            fbqueue->release(fb);

            if (!result) {
                continue;
            }

            uint8_t *pixels = decoder.getOutputFrame();
            size_t width = decoder.getOutputWidth(), height = decoder.getOutputHeight();

            // Find line with maximum of the desired color
            size_t maxy = 0, maxymatches = 0;

            for (size_t y = 0; y < height; y++) {
                size_t nmatches = 0;

                for (size_t x = 0; x < width; x++) {
                    uint8_t *pixel = pixels + (y * width + x) * 3;
                    HsvColor hsv = RgbToHsv({pixel[0], pixel[1], pixel[2]});

                    if (abs(_color.h - hsv.h) < HUE_THRESHOLD && abs(_color.s - hsv.s) < SATURATION_THRESHOLD) {
                        nmatches++;
                    }
                }

                if (maxymatches < nmatches) {
                    maxymatches = nmatches;
                    maxy = y;
                }
            }


            // Find column with maximum of the desired color
            size_t maxx = 0, maxxmatches = 0;

            for (size_t x = 0; x < width; x++) {
                size_t nmatches = 0;

                for (size_t y = 0; y < height; y++) {
                    uint8_t *pixel = pixels + (y * width + x) * 3;
                    HsvColor hsv = RgbToHsv({pixel[0], pixel[1], pixel[2]});

                    if (abs((int)_color.h - (int)hsv.h) < HUE_THRESHOLD && abs((int)_color.s - (int)hsv.s) < SATURATION_THRESHOLD) {
                        nmatches++;
                    }
                }

                if (maxxmatches < nmatches) {
                    maxxmatches = nmatches;
                    maxx = x;
                }
            }

            if (maxx && maxy) {
                /*
                Serial.print("Found blob at ");
                Serial.print((float)maxx / width);
                Serial.print("x");
                Serial.println((float)maxy / height);
                */

                _detected = {(float)maxx / width, (float)maxy / height, 0.2, 0.2};
                xSemaphoreGive(_signal);
            }

            _framerate.tick();
        }
    }
}

void BlobDetector::runStatic(void *p) {
    BlobDetector *detector = (BlobDetector *)p;
    detector->run();
    vTaskDelete(NULL);
}
