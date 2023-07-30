PYTHON=python3

venv:
	$(PYTHON) -m venv venv && . ./venv/bin/activate && pip install -r requirements.txt || rm -rf venv

clean:
	rm -rf venv