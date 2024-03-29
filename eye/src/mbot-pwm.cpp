#include "mbot-vision/mbot-pwm.h"

#include <Arduino.h>
#include "mbot-vision/detection/detector.h"
#include "mbot-vision/common.h"

#define MV_PWM_FREQ    1024
#define MV_PWM_BITS    10       // 10 bits gives (1 << 10) = 1024 values
#define MV_PWM_OFF     32       // Frequency indicating no detection
#define MV_PWM_MIN     256
#define MV_PWM_MAX     1024 
#define MV_PWM_RANGE   (MV_PWM_MAX - MV_PWM_MIN)

MBotPWM::MBotPWM(Detector &detector)
 : _detector(detector) {}

void MBotPWM::begin() {
    xTaskCreatePinnedToCore(runStatic, "mBotPWM", 10000, this, 2, &_task, 1);
}

void MBotPWM::run() {
    Serial.println("Starting PWM comms with MBot");
    ledcAttachPin(MV_PWMX_PIN, MV_PWMX_CHAN);
    ledcAttachPin(MV_PWMY_PIN, MV_PWMY_CHAN);

    ledcSetup(MV_PWMX_CHAN, MV_PWM_FREQ, MV_PWM_BITS);
    ledcSetup(MV_PWMY_CHAN, MV_PWM_FREQ, MV_PWM_BITS);

    while (true) {
        DetectedObject blob = _detector.get();
        
        if (blob.detected) {
            ledcWrite(MV_PWMX_CHAN, MV_PWM_MIN + (int)((float)MV_PWM_RANGE * blob.x));
            ledcWrite(MV_PWMY_CHAN, MV_PWM_MIN + (int)((float)MV_PWM_RANGE * blob.y));
            //ledcWrite(MV_PWMY_CHAN, (1 << (MV_PWM_BITS - 1)) + (int)((float)hsv.h * (360.0f / 256.0f)));
        }
        else {
            ledcWrite(MV_PWMX_CHAN, MV_PWM_OFF);
            ledcWrite(MV_PWMY_CHAN, MV_PWM_OFF);
        }
 
        delay(50);
    }
}

void MBotPWM::runStatic(void *p) {
    MBotPWM *mbot = (MBotPWM *)p;
    mbot->run();
    vTaskDelete(NULL);
}
