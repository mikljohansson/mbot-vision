#ifndef _MV_HTTPD_H_
#define _MV_HTTPD_H_

#include <Arduino.h>
#include <esp_camera.h>

void httpdInit();
void httpdLoop();
void httpdSendStream(camera_fb_t *fb);

#endif
