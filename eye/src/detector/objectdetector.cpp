#include <mbot-vision-model.h>

#ifdef MBOT_VISION_INFERENCE_ENGINE_TFMICRO
    #include "objectdetector-tfmicro.h"
#else
    #error No inference engine defined
#endif
