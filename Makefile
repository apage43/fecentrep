PYTHON=python3

env:
	$(PYTHON) -m venv env && . ./env/bin/activate && pip install -r requirements.txt || rm -rf env

clean:
	rm -rf env