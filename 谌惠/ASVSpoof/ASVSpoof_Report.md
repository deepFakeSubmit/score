# Spoof Speech Recognition

Here we have two systems which are designed for spoof speech recognition,one uses
* Mel-frequency cepstrum coefficients (MFCC)
* Gaussian mixture models (GMM)

while the other does the same using
* RawNet2

## Dataset
The *zju_deepfake dataset* consists of three independent partitions: train, development and evaluation.
The training and development partitions contain bonafide and spoofed data with labels,
whereas the evaluation partition contains bona fide speech and spoofed data without.

## MFCC-GMM

### Reference
[voice-based-gender-recognition](https://github.com/SuperKogito/Voice-based-gender-recognition) for extracting features and training the Gaussian Mixture Models.

Library *spafe* is used here instead of *python_speech_features* for voice features extraction.

### Dependencies
This script require the follwing modules/libraries:
* numpy & sklearn & spafe & soundfile

### Code & Scripts
* [run.py](run.py):This is the main script and it will run the whole cycle (Models training > Spoof speech recognition)
* [Identifier.py](Identifier.py):This script is responsible for Testing the system by Spoof Speech Recognition of the testing set (dev and eval).
* [ModelsTrainer.py](ModelsTrainer.py):This script is responsible for training the Gaussian Mixture Models (GMM).
* [MFCC_FeaturesExtractor.py](MFCC_FeaturesExtractor.py):This script is responsible for extracting the MFCC features from the .flac files.

### Adjustment
* Speech feature processing
    * None
    ```
        Accuracy = 81.16%
        Recall = 95.736% 
        Precision = 34.936% 
    ```
    * CMS
    ```
        Accuracy = 83.2%
        Recall = 98.062%
        Precision = 37.874%
    ```
    * CMS + MFCC delta & MFCC double del
    ```
        Accuracy = 82.52%
        Recall = 98.062%
        Precision = 36.934%
    ```
* Parameters of Gaussian Mixture Model
    * n_components = 32
    ```
        Accuracy = 83.52%
        Recall = 99.225%
        Precision = 38.438%
    ```
    * n_components = 64
    ```
        Accuracy = 84.16%
        Recall = 98.837%
        Precision = 39.352%
    ```
    * max_iter = 10
    ```
        Accuracy = 82.52%
        Recall = 98.837%
        Precision = 37.01%
    ```  
    * max_iter = 400
    ```
    Accuracy = 84.08% ***
    Recall = 99.225% ***
    Precision = 39.264% ***
    ```
    
## RawNet2

### Reference
[RawNet2 ASVspoof 2021 baseline](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2):using
an end-to-end method that uses a model based on the RawNet2 topology as described the paper accepted to ICASSP 2021, ["End-to-end anti-spoofing with RawNet2"](https://arxiv.org/abs/2011.01108).

### Dependencies
Install the requirements :
```
pip install -r requirements.txt
```

### Notice
1. Change `database_path` in [main.py](Baseline-RawNet2\main.py) to your local database path.
2. Change `protocols_path` in [main.py](Baseline-RawNet2\main.py) to database protocols directory address
3. Confirm the format of your protocols files.The format that works here is `filename label`.You can modify `genSpoof_list` in [data_utils](Baseline-RawNet2\data_utils.py) to fit your format.

### Training
To train the model run:
```shell script
python main.py --loss=CCE --lr=0.0001 --batch_size=32
```

### Testing
```shell script
python main.py --loss=CCE --is_eval --eval --model_path='/path/to/your/your_best_model.pth' --eval_output='eval_CM_labels.txt'
```