#include "colorspace.h"

// https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
RgbColor HsvToRgb(HsvColor hsv)
{
    RgbColor rgb;
    int region, p, q, t;
    int h, s, v, remainder;

    if (hsv.s == 0)
    {
        rgb.r = hsv.v;
        rgb.g = hsv.v;
        rgb.b = hsv.v;
        return rgb;
    }

    // converting to 16 bit to prevent overflow
    h = hsv.h;
    s = hsv.s;
    v = hsv.v;

    region = h / 43;
    remainder = (h - (region * 43)) * 6; 

    p = (v * (255 - s)) >> 8;
    q = (v * (255 - ((s * remainder) >> 8))) >> 8;
    t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
        case 0:
            rgb.r = v;
            rgb.g = t;
            rgb.b = p;
            break;
        case 1:
            rgb.r = q;
            rgb.g = v;
            rgb.b = p;
            break;
        case 2:
            rgb.r = p;
            rgb.g = v;
            rgb.b = t;
            break;
        case 3:
            rgb.r = p;
            rgb.g = q;
            rgb.b = v;
            break;
        case 4:
            rgb.r = t;
            rgb.g = p;
            rgb.b = v;
            break;
        default:
            rgb.r = v;
            rgb.g = p;
            rgb.b = q;
            break;
    }

    return rgb;
}

HsvColor RgbToHsv(RgbColor rgb)
{
    HsvColor hsv;
    int rgbMin, rgbMax;

    rgbMin = min(rgb.r, min(rgb.g, rgb.b));
    rgbMax = max(rgb.r, max(rgb.g, rgb.b));

    hsv.v = rgbMax;
    if (hsv.v == 0)
    {
        hsv.h = 0;
        hsv.s = 0;
        return hsv;
    }

    hsv.s = 255 * (rgbMax - rgbMin) / hsv.v;
    if (hsv.s == 0)
    {
        hsv.h = 0;
        return hsv;
    }

    if (rgbMax == rgb.r)
        hsv.h = 0 + 43 * ((int)rgb.g - (int)rgb.b) / (rgbMax - rgbMin);
    else if (rgbMax == rgb.g)
        hsv.h = 85 + 43 * ((int)rgb.b - (int)rgb.r) / (rgbMax - rgbMin);
    else
        hsv.h = 171 + 43 * ((int)rgb.r - (int)rgb.g) / (rgbMax - rgbMin);

    return hsv;
}
