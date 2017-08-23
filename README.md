# MatchRAM
A Recurrent Attention Model for Matching Two Images

## Environment
* Pull the docker image to containerize the training. For cpu:
    ```bash
    $ docker pull siavashk/siemens:cpu
    ```
    For gpu:
    ```bash
    $ docker pull siavashk/siemens:gpu
    ```
* Start the notebook by running either `startcpu.sh` or `startgpu.sh`

## Data
* Download the [affNIST](http://www.cs.toronto.edu/~tijmen/affNIST/) dataset.
* Extract the contents to a directory of your choice. Recommended path is `data/affNIST`
* Create a paired dataset by running:
    ```bash
    $ python scripts/utilities/makePairs.py
    ```
* [Optional] Verify that the data was properly created by running:
    ```bash
    $ python scripts/utilities/testPairs.py
    ```

## Training
* Open a terminal from the notebook homepage
* [Optional] Modify training parameters in `config/config.py`
* Start the training by running `python scripts/train.py`
