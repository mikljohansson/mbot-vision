#include <mbot-vision-model.h>

#ifdef MBOT_VISION_INFERENCE_TFMICRO
    #include "objectdetector-tfmicro.h"
#elif MBOT_VISION_INFERENCE_TFMICRO_LEGACY
    // Older version of tflite-micro which seems to have 2x better performance
    #include "objectdetector-tfmicro-legacy.h"
#elif MBOT_VISION_INFERENCE_TINYMAIX
    #include "objectdetector-tinymaix.h"
#else
    #error No inference engine defined
#endif
