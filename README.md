Create a virtual environment: `python3 -m venv env`. Activate it with `source 
env/bin/activate`. Then, `pip install -r requirements.txt`. 

Run `ipython kernel install --user`. This is because sometimes the path points 
to the root instead of the virtual environment. To check, open the notebook and 
run: 
```python 
import sys 
sys.path
sys.executable
```
to make sure it's pointing to the virtual environment. 
