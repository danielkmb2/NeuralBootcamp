# NEURAL BOOTCAMP

Json based neural networks builder. This project aims on defining an easy way to build simple neural networks based on
dense and recurrent layers, with automatic data preprocessing and some utils for infering new samples.

```bash
# train model
python run.py /data/configuration_file.json

# prediction only example
python prediction_only_example.py sample_configs/config_multiparams.json saved_models/config_multiparams-09102018-185633.h5
```

* See examples in `/sample_configs`

## Requirements

Install requirements.txt file to make sure correct versions of libraries are being used.

* Python 3.6.x
* TensorFlow 1.10.0
* Numpy 1.15.0
* Keras 2.2.2
* Matplotlib 2.2.2
* Pandas 0.23.3

## To-Do

* GRU support
* Multiple input layers
* Grid search optimizations
* Cross-validation for recurrent-based problems