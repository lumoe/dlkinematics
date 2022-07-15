[![.github/workflows/main.yml](https://github.com/lumoe/dlkinematics/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/lumoe/dlkinematics/actions/workflows/main.yml)

# Deep Learning Kinematics

### Differentiable Forwad Kinematics for TensorFlow and Keras

Supported Joint Types:

- [x] Fixed
- [x] Revolute
- [x] Continious
- [x] Prismatic
- [x] Floating (not coverd by unit tests)
- [x] Planar (not coverd by unit tests)

## Installation

### Install from source

`$ pip3 install -e git+https://github.com/lumoe/dlkinematics.git@main#egg=DLKinematics`

### Install from PyPi

_Coming soon_

## Usage:

```python
import tensorflow as tf
from dlkinematics.urdf import chain_from_urdf_file
from dlkinematics.dlkinematics import DLKinematics

# Load URDF
chain = chain_from_urdf_file('data/human.urdf')

# Create DLKinematics
dlkinematics = DLKinematics(
   chain,
   base_link="human_base",
   end_link="human_spine_2",
   batch_size=2)

# Joint configuartion
thetas = tf.Variable([1., 2., 3., 4.], dtype=tf.float32)

# Forward pass
with tf.GradientTape() as tape:
    result = dlkinematics.forward(thetas)

print(result)
print(tape.gradient(result, thetas))

```

## As Keras Layer

```python
from dlkinematics.training_utils import ForwardKinematics

model = keras.Sequential()

FK_layer = ForwardKinematics(
   urdf = 'path/to/urdf',
   base_link = 'link0',
   end_link = 'linkN',
   batch_size = 32)

model.add(FK_layer)
# Output shape of FK_layer is (batch_size, 4, 4)
```

## Run tests

The tests use ROS packages to validate the result of the dlkinematics module.

1. Build the docker image for tests:  
   `$ docker build -t dlkinematics_tests .`

1. Start the container in the root folder of the project:  
   `$ docker run -it -v $PWD:/work dlkinematics_tests python3 -m pytest`

1. Execute all tests:  
   `$ docker run -it -v $PWD:/work dlkinematics_tests python3 -m pytest`  
   Execute only a single testfile:  
   `$ docker run -it -v $PWD:/work dlkinematics_tests python3 -m pytest tests/test_prismatic.py`
