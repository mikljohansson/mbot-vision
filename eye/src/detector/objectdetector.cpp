#include <mbot-vision-model.h>

#ifdef MBOT_VISION_INFERENCE_TFMICRO
    #include "objectdetector-tfmicro.h"
#elif MBOT_VISION_INFERENCE_TINYMAIX
    #include "objectdetector-tinymaix.h"
#else
    #error No inference engine defined
#endif
