#include "mbot-serial.h"

void MBotSerial::start() {
    xTaskCreatePinnedToCore(runStatic, "mbotSerial", 10000, this, 1, &_task, 0);
}

void MBotSerial::run() {
    Serial.println("Starting receive from MBot");
    HardwareSerial &uart = Serial1;
    uart.setDebugOutput(true);
    uart.begin(115200, SERIAL_8N1, MV_URX_PIN, MV_UTX_PIN);

    while (true) {
        String line;
        int c = 0;

        while (c != '\n') {
            delay(1000);
            Serial.println("no serial data available");

            while (uart.available() > 0) {
                c = uart.read();
                if (c == '\n') {
                    break;
                }

                Serial.print("fragment: ");
                Serial.println(line);
                line += (char)c;
            }
        }

        Serial.print("line: ");
        Serial.println(line);
    }
}

void MBotSerial::runStatic(void *p) {
    MBotSerial *mbot = (MBotSerial *)p;
    mbot->run();
    vTaskDelete(NULL);
}
