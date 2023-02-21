#ifndef _MV_FRAMERATE_H_
#define _MV_FRAMERATE_H_

class Framerate {
    private:
        const char *_format;
        double windowTime = 0.0;
        unsigned long windowFrames = 0;
        unsigned long lastUpdatedWindow;
        unsigned long lastShowedFramerate = 0;

    public:
        Framerate(const char *format);
        void init();
        void tick();
};

#endif