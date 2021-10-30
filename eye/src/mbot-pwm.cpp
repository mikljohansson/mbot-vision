#include <Arduino.h>
#include "blobdetector.h"
#include "mbot-pwm.h"

#define MV_PWM_FREQ    1000
#define MV_PWM_BITS    13   // Resolution 8, 10, 12, 15
#define MV_PWM_OFF     (1 << (MV_PWM_BITS - 8))
#define MV_PWM_MIN     (1 << (MV_PWM_BITS - 6))
#define MV_PWM_MAX     ((1 << MV_PWM_BITS) - MV_PWM_MIN)
#define MV_PWM_RANGE   (MV_PWM_MAX - MV_PWM_MIN)

MBotPWM::MBotPWM(BlobDetector &detector)
 : _detector(detector) {}

void MBotPWM::start() {
    xTaskCreatePinnedToCore(runStatic, "mbot-pwm", 10000, this, 1, &_task, 0);
}

void MBotPWM::run() {
    Serial.println("Starting PWM comms with MBot");
    ledcAttachPin(MV_PWMX_PIN, MV_PWMX_CHAN);
    ledcAttachPin(MV_PWMY_PIN, MV_PWMY_CHAN);

    ledcSetup(MV_PWMX_CHAN, MV_PWM_FREQ, MV_PWM_BITS);
    ledcSetup(MV_PWMY_CHAN, MV_PWM_FREQ, MV_PWM_BITS);

    while (true) {
        DetectedBlob blob = _detector.get();
        
        if (blob.detected) {
            ledcWrite(MV_PWMX_CHAN, MV_PWM_MIN + (int)((float)MV_PWM_RANGE * blob.x));
            ledcWrite(MV_PWMY_CHAN, MV_PWM_MIN + (int)((float)MV_PWM_RANGE * (1.0f - blob.y)));
        }
        else {
            ledcWrite(MV_PWMX_CHAN, MV_PWM_OFF);
            ledcWrite(MV_PWMY_CHAN, MV_PWM_OFF);
        }
 
        delay(25);
    }
}

void MBotPWM::runStatic(void *p) {
    MBotPWM *mbot = (MBotPWM *)p;
    mbot->run();
    vTaskDelete(NULL);
}
