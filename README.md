# 02461-Exam-Project

# Virtual Environment

I dette projekt gør vi brug af et virtuelt miljø med Python 3.11.7, for at sikre kompatibilitet mellem styresystemer. For at opsætte dette virtuelle miljø kan du med `Command Palette` i VSCode (CTRL+SHIFT+P) køre
``Python: Create Environment`` og derefter vælge enten venv eller Conda (vi har brugt venv).
Du kan installere alle nødvendige pakker med `py -m pip install -r requirements.txt`.  

Tjek installation med 

```python
python -c "import os, sys; print(os.path.dirname(sys.executable))"
```
