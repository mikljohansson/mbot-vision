import os, sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

width = 160
height = 120

def representative_data_gen():
    found = 0
    samples = 10
    dn = sys.argv[3]

    for fn in os.listdir(dn):
        if not fn.endswith('.jpg'):
            continue

        print("Reading calibration image {}".format(fn))
        fn = os.path.join(dn, fn)
        img = image.load_img(fn, target_size=(width, height))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        found += 1
        yield [x]

        if found >= samples:
            break

    if found < samples:
        raise ValueError("Failed to read %d calibration sample images" % samples)

BLOCKS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
#    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
#     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
#    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
#     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

if __name__ == '__main__':
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet
    model = tf.keras.applications.EfficientNetB0(
                include_top=True, 
                weights=None,
                input_shape=(width, height, 3), 
                classes=10,
                blocks_args=BLOCKS)
    model.compile()
    model.load_weights(sys.argv[1], by_name=True, skip_mismatch=True)
    model.summary()

    #new_model = change_model(model)
    #new_model.summary()

#    for inp in representative_data_gen():
 #       pred = model.predict(inp)
  #      dpred = decode_predictions(pred)
   #     print([p for p in dpred[0] if p[2] > 0.5])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    open(sys.argv[2], 'wb').write(tflite_model)
