#ifndef _MV_DATALOG_H_
#define _MV_DATALOG_H_

#include <Arduino.h>

class DataLogger {
    private:
        String _directory;
        bool _enabled;
        bool _active;
        bool _initialized;
        time_t _lastLogged;
    
    public:
        DataLogger();
        bool isEnabled();
        bool isActive();
        void setActive(bool active);
        bool begin();
        void logJpeg(const uint8_t *data, size_t len);
};

#endif
