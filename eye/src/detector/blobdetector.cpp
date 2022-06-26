#include "detection/blobdetector.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "image/colorspace.h"

#define HUE_THRESHOLD           16
#define SATURATION_THRESHOLD    64
#define LUMINOSITY_THRESHOLD    128
#define MIN_BLOB_WIDTH          3
#define MIN_BLOB_HEIGHT         2

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
    char colorBuffer[24];
    snprintf(colorBuffer, sizeof(colorBuffer), "#%02X%02X%02X", color.r, color.g, color.b);

    out += "{\"x\":";
    out += x;
    out += ",\"y\":";
    out += y;
    out += ",\"detected\":";
    out += detected;
    out += ",\"color\":\"";
    out += colorBuffer;
    out += "\"}";
}

BlobDetector::BlobDetector(RgbColor color)
 : _color(color), _framerate("Blob detector framerate: %02f\n"), _detected({0, 0, false}) {
    _signal = xSemaphoreCreateBinary();
}

BlobDetector::~BlobDetector() {
    vSemaphoreDelete(_signal);
}

void BlobDetector::start() {
    xTaskCreatePinnedToCore(runStatic, "blobDetector", 10000, this, 1, &_task, 1);
}

DetectedBlob BlobDetector::wait() {
    xSemaphoreTake(_signal, portMAX_DELAY);
    return get();
}

DetectedBlob BlobDetector::get() {
    return _detected;
}

void BlobDetector::draw(uint8_t *pixels, size_t width, size_t height) {
    size_t framelen = width * height * 3;
    HsvColor hsv = RgbToHsv(_color);
    
    /*
    Serial.print("Top pix ");
    printPixel(pixels[0], pixels[1], pixels[2]);
    */

    int i = 0;
    for (uint8_t *pixel = pixels, *last = pixel + framelen; pixel < last; pixel += 3) {
        if (isColorMatch(hsv, pixel)) {
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
        else {
            //std::swap(pixel[0], pixel[2]);

            // Convert to HSV and back
            RgbColor rgb = HsvToRgb(RgbToHsv({pixel[0], pixel[1], pixel[2]}));

            // Something seems to flip the red and blue channels
            pixel[0] = rgb.b;
            pixel[1] = rgb.g;
            pixel[2] = rgb.r;
        }

        if (i++ % width == 0) {
            yield();
        }
    }
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

                if (maxymatches < nmatches && nmatches >= MIN_BLOB_WIDTH) {
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

                if (maxxmatches < nmatches && nmatches >= MIN_BLOB_HEIGHT) {
                    maxxmatches = nmatches;
                    maxx = x;
                }

                yield();
            }

            if (maxx && maxy) {
                /*
                Serial.print("Found color ");
                const uint8_t *pixel = pixels + (maxy * width + maxx) * 3;
                printPixel(pixel[0], pixel[1], pixel[2]);
                */

                //uint8_t *pixel = pixels + (maxy * width + maxx) * 3;
                //RgbColor rgb = {pixel[0], pixel[1], pixel[2]};
                _detected = {(float)maxx / width, (float)maxy / height, true, _color};
                xSemaphoreGive(_signal);
            }
            else {
                _detected = {0, 0, false, {0, 0, 0}};
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
