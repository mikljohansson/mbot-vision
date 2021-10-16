#include "blobdetector.h"
#include "camera.h"
#include "jpeg.h"
#include "colorspace.h"

#define HUE_THRESHOLD           60
#define SATURATION_THRESHOLD    192
#define LUMINOSITY_THRESHOLD    192

static inline bool isColorMatch(const HsvColor &color, const uint8_t *pixel) {
    HsvColor hsv = RgbToHsv({pixel[0], pixel[1], pixel[2]});
    return abs((((int)color.h - (int)hsv.h) + 128) % 256 - 128) < HUE_THRESHOLD 
        && abs((int)color.s - (int)hsv.s) < SATURATION_THRESHOLD
        && abs((int)color.v - (int)hsv.v) < LUMINOSITY_THRESHOLD;
}

void printPixel(uint8_t r, uint8_t g, uint8_t b) {
    HsvColor hsv = RgbToHsv({r, g, b});

    Serial.print("rgb: ");
    Serial.print(r);
    Serial.print(",");
    Serial.print(g);
    Serial.print(",");
    Serial.print(b);
    Serial.print(" hsv: ");
    Serial.print(hsv.h);
    Serial.print(",");
    Serial.print(hsv.s);
    Serial.print(",");
    Serial.println(hsv.v);
}

void DetectedBlob::serialize(String &out) {
    out += "{\"x\":";
    out += x;
    out += ",\"y\":";
    out += y;
    out += "}";
}

BlobDetector::BlobDetector(RgbColor color)
 : _color(color), _framerate("Blob detector framerate: %02f\n") {
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

    Serial.print("Starting blob detector for ");
    printPixel(_color.r, _color.g, _color.b);

    HsvColor hsv = RgbToHsv(_color);

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
                    if (isColorMatch(hsv, pixels + (y * width + x) * 3)) {
                        nmatches++;
                    }
                }

                if (maxymatches < nmatches) {
                    maxymatches = nmatches;
                    maxy = y;
                }

                yield();
            }

            // Find column with maximum of the desired color
            size_t maxx = 0, maxxmatches = 0;

            for (size_t x = 0; x < width; x++) {
                size_t nmatches = 0;

                for (size_t y = 0; y < height; y++) {
                    if (isColorMatch(hsv, pixels + (y * width + x) * 3)) {
                        nmatches++;
                    }
                }

                if (maxxmatches < nmatches) {
                    maxxmatches = nmatches;
                    maxx = x;
                }

                yield();
            }

            if (maxx && maxy) {
                Serial.print("Found color ");
                const uint8_t *pixel = pixels + (maxy * width + maxx) * 3;
                printPixel(pixel[0], pixel[1], pixel[2]);

                _detected = {(float)maxx / width, (float)maxy / height};
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
