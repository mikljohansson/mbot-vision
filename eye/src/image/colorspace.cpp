#include "image/colorspace.h"

static inline uint8_t fhr(HsvColor hsv, float n) {
    float k = fmod(n + (float)hsv.h / 255.0f * 360.0f / 60.0f, 6.0f);
    return hsv.v - (uint8_t)((float)hsv.v * (float)hsv.s / 255.0f * max(min(min(k, 4.0f - k), 1.0f), 0.0f));
}

// https://stackoverflow.com/questions/17242144/javascript-convert-hsb-hsv-color-to-rgb-accurately/54024653#54024653
RgbColor HsvToRgb(HsvColor hsv) {
    return {fhr(hsv, 5.0f), fhr(hsv, 3.0f), fhr(hsv, 1.0f)};
}

// https://stackoverflow.com/questions/8022885/rgb-to-hsv-color-in-javascript/54070620#54070620
HsvColor RgbToHsv(RgbColor rgb) {
    uint8_t v = max(rgb.r, max(rgb.g, rgb.b));
    float vf = (float)v / 255.0f;

    uint8_t c = v - min(rgb.r, min(rgb.g, rgb.b));
    float cf = (float)c / 255.0f;

    float h = c
        ? ((v == rgb.r) 
            ? ((float)(rgb.g - rgb.b) / 255.0f / cf)
            : ((v == rgb.g) 
                ? (2.0f + (float)(rgb.b - rgb.r) / 255.0f / cf) 
                : (4.0f + (float)(rgb.r - rgb.g) / 255.0f / cf)))
        : 0.0f;

    return {(uint8_t)(60.0 * (h < 0.0f ? h + 6.0f : h) / 360.0f * 255.0f), v ? (uint8_t)(cf / vf * 255.0f) : (uint8_t)0, v};
}
