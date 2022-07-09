
#include "detection/detector.h"

void DetectedObject::serialize(String &out) {
    out += "{\"x\":";
    out += x;
    out += ",\"y\":";
    out += y;
    out += ",\"detected\":";
    out += detected;
    out += "}";
}