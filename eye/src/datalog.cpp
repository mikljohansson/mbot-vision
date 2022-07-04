#include "datalog.h"
#include "wiring.h"
#include <FS.h>
#include <SD_MMC.h>
#include <TimeLib.h>

#define DIRECTORY_FORMAT "/sdcard/mbot-vision-%d%02d%02d-%02d%02d"

DataLogger::DataLogger()
 : _enabled(false), _lastLogged(0) {}

bool DataLogger::begin() {
    int len = snprintf(NULL, 0, DIRECTORY_FORMAT, year(), month(), day(), hour(), minute());
    if (len) {
        char buf[len];
        sprintf(buf, DIRECTORY_FORMAT, DIRECTORY_FORMAT, year(), month(), day(), hour(), minute());
        _directory = buf;
    }
    
    //SD_MMC.setPins(MV_HS2_CLK, MV_HS2_CMD, MV_HS2_DATA0, MV_HS2_DATA1, MV_HS2_DATA2, MV_HS2_DATA3);
    SD_MMC.setPins(MV_HS2_CLK, MV_HS2_CMD, MV_HS2_DATA0);
    _enabled = SD_MMC.begin();

    if (!_enabled) {
        Serial.println("Failed to initialize SD card");
    }

    return _enabled;
}

void DataLogger::logJpeg(const uint8_t *data, size_t length) {
    unsigned long ts = millis();
    
    if (!_enabled || _lastLogged >= (ts - 1000)) {
        return;
    }

    _lastLogged = ts;

    String filename = _directory + "/" + now() + ".jpg";
    File file = SD_MMC.open(filename.c_str(), FILE_WRITE, true);
    size_t written = file.write(data, length);

    if (written < length) {
        Serial.println("Failed to write to SD card");
    }

    file.close();
}
