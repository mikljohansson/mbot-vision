#ifndef _MV_COLORSPACE_H_
#define _MV_COLORSPACE_H_

#include <Arduino.h>

// https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
typedef struct RgbColor
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RgbColor;

typedef struct HsvColor
{
    uint8_t h;
    uint8_t s;
    uint8_t v;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv);

HsvColor RgbToHsv(RgbColor rgb);

#endif