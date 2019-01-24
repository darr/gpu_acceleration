env_path=~/.pythorch_env
#   source $env_path/py2env/bin/activate
#   pylint --rcfile=pylint.conf main.py
#   pip freeze > python2_requiements.txt
#   python main.py
#   deactivate

source $env_path/py3env/bin/activate
pylint --rcfile=pylint.conf main.py
#pip freeze > python3_requiements.txt
python main.py
deactivate
