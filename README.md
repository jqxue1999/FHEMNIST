# FHEMNIST

## requirements
- python>=3.8.12
- torch>=1.13.1
- torchvision>=0.14.1
- numpy>=1.23.5
- tenseal>=0.3.14
- psutil>=5.9.0
- tqdm>=4.64.1

## train the model
```bash
python -u train.py
```
Then you will get the model file  at `./checkpoint/best.pt`

## encrypt the model and evaluate
```bash
python -u fhe.py
```
Then you will get the accuracy of the encrypted model.
