#include <Arduino.h>
#include <MeAuriga.h>

// Definitions copied from https://github.com/Makeblock-official/Makeblock-Libraries/blob/master/examples/Firmware_for_Auriga/Firmware_for_Auriga.ino
#define RGBLED_PORT                          44
MeRGBLed led;

void setup() {
    led.setpin(RGBLED_PORT);
    led.setNumber(12);
    led.setColor(0, 0, 0, 0);
    led.show();

    Serial.begin(115200);
    Serial.println("Starting up");
    Serial2.begin(115200);
}

long tsLast = 0;
long count = 0;

void loop() {
    int c;
    do {
        c = Serial2.read();
        if (c > 0) {
            char cc = c;
            Serial.write(&cc, 1);
        }
    } while (c > 0);

    if (millis() - tsLast > 1000) {
        tsLast = millis();
        Serial2.print("hi from mbot");
    }
}
