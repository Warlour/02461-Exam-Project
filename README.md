# 02461-Exam-Project

## Virtual Environment

I dette projekt gør vi brug af et virtuelt miljø med Python 3.11.7, for at sikre kompatibilitet mellem styresystemer. For at opsætte dette virtuelle miljø kan du med `Command Palette` i VSCode (CTRL+SHIFT+P) køre
``Python: Create Environment`` og derefter vælge enten venv eller Conda (vi har brugt venv).
Du kan installere alle nødvendige pakker med `py -m pip install -r requirements.txt`.

Tjek installation med

```python
py -c "import os, sys; print(os.path.dirname(sys.executable))"
```

## Cuda

For at kunne udnytte GPU'en til træning af modellen skal du hente en cuda understøttet version af PyTorch. Dette kan gøres ved at gå ind på [PyTorch](https://pytorch.org/get-started/locally/) og vælge den rigtige version.

## Usage

Brug `-h` argumentet for at få en liste over alle argumenter.

# Model files

The .pt files are to be saved with the following format: `date time batchsize-learningrate-epochs-weightdecay-gamma-min_lr-momentum accuracy lossfunc-optifunc-scheduler note.pt`
