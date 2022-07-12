from setuptools import setup

# Requirements get automatically synced via `pipenv-setup sync`
setup(install_requires=["absl-py==1.1.0; python_version >= '3.6'", 'astunparse==1.6.3', "cachetools==5.2.0; python_version ~= '3.7'", "certifi==2022.6.15; python_version >= '3.6'", "charset-normalizer==2.1.0; python_version >= '3.6'", "cycler==0.11.0; python_version >= '3.6'", "dill==0.3.5.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6'", "etils[epath]==0.6.0; python_version >= '3.7'", 'flatbuffers==1.12', "fonttools==4.34.4; python_version >= '3.7'", "gast==0.4.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "google-auth==2.9.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'", "google-auth-oauthlib==0.4.6; python_version >= '3.6'", 'google-pasta==0.2.0', "googleapis-common-protos==1.56.3; python_version >= '3.6'", "grpcio==1.47.0; python_version >= '3.6'", "h5py==3.7.0; python_version >= '3.7'", "idna==3.3; python_version >= '3.5'", "importlib-metadata==4.12.0; python_version < '3.10'", "importlib-resources==5.8.0; python_version < '3.9'", 'keras==2.9.0', 'keras-preprocessing==1.1.2', "kiwisolver==1.4.3; python_version >= '3.7'", 'libclang==14.0.1', 'lxml==4.9.1', "markdown==3.3.7; python_version >= '3.6'", "matplotlib==3.5.2; python_version >= '3.7'", "networkx==2.8.4; python_version >= '3.8'", 'numpy==1.23.1', "oauthlib==3.2.0; python_version >= '3.6'", 'openexr==1.3.8', "opt-einsum==3.3.0; python_version >= '3.5'", "packaging==21.3; python_version >= '3.6'", 'pandas==1.4.3', "pillow==9.2.0; python_version >= '3.7'", 'progressbar2==4.0.0', 'promise==2.3', "protobuf==3.19.4; python_version >= '3.5'", "psutil==5.9.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", 'pyasn1==0.4.8', 'pyasn1-modules==0.2.8', "pyparsing==3.0.9; python_full_version >= '3.6.8'", "python-dateutil==2.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "python-utils==3.3.3; python_version >= '3.7'", 'pytz==2022.1', 'pyyaml==6.0', "requests==2.28.1; python_version >= '3.7' and python_version < '4'", "requests-oauthlib==1.3.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "rsa==4.8; python_version >= '3.6'", "scipy==1.8.1; python_version < '3.11' and python_version >= '3.8'", "setuptools==63.1.0; python_version >= '3.7'", "six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", "tensorboard==2.9.1; python_version >= '3.6'", "tensorboard-data-server==0.6.1; python_version >= '3.6'", 'tensorboard-plugin-wit==1.8.1', 'tensorflow==2.9.1', 'tensorflow-addons==0.17.1', "tensorflow-datasets==4.6.0; python_version >= '3.7'", "tensorflow-estimator==2.9.0; python_version >= '3.7'", 'tensorflow-gpu==2.9.1', 'tensorflow-graphics==2021.12.3', "tensorflow-io-gcs-filesystem==0.26.0; python_version < '3.11' and python_version >= '3.7'", "tensorflow-metadata==1.9.0; python_version >= '3.7' and python_version < '4'", 'termcolor==1.1.0', "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'", "tqdm==4.64.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'", 'transformations==2021.6.6', 'trimesh==3.12.7', "typeguard==2.13.3; python_full_version >= '3.5.3'", "typing-extensions==4.3.0; python_version >= '3.7'", "urllib3==1.26.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5' and python_version < '4'", "werkzeug==2.1.2; python_version >= '3.7'", "wheel==0.37.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", "wrapt==1.14.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'", 'zipp==3.8.1'],name='DLKinematics',
      python_requires='>=3.8',
      version='0.0.1',
      description='Differentiable Forward Kinematics for TensorFlow and Keras based on URDF files',
      url='https://github.com/lumoe/dlkinematics',
      author='Lukas Mölschl',
      license='BSD 3-Clause',
      packages=['dlkinematics'],
      zip_safe=False)