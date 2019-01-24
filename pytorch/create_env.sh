#Assuming that you have install vitralenv,python2,python3.
env_path=~/.pythorch_env
rm -rf $env_path
mkdir $env_path
cd $env_path
virtualenv -p /usr/bin/python2 py2env
source $env_path/py2env/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade matplotlib==2.2.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torchvision

deactivate
virtualenv -p /usr/bin/python3 py3env
source $env_path/py3env/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torchvision
deactivate
