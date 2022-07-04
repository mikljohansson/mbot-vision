#ifndef _MV_MBOT_DATALOG_H_
#define _MV_MBOT_DATALOG_H_

#include <Arduino.h>

class DataLogger {
    private:
        String _directory;
        bool _enabled;
        time_t _lastLogged;
    
    public:
        DataLogger();
        bool begin();
        void logJpeg(const uint8_t *data, size_t len);
};

#endif
