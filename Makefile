VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit

# include .env file and export env variables
include .env
export

# Need to use python 3.9 for aws lambda
$(VENV)/bin/activate: requirements.txt
	python3.12 -m venv $(VENV)
	$(PIP) install -r requirements.txt


app: $(VENV)/bin/activate
	$(STREAMLIT) run app.py

fc: $(VENV)/bin/activate
	$(PYTHON) fc.py

test: $(VENV)/bin/activate
	$(PYTHON) -m unittest test.py

u2s: $(VENV)/bin/activate
	$(PYTHON) un2structured.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)