echo [$(date)]: "START"
echo [$(date)]: "Creating env with python 3.12 version"
conda create --prefix ./env python=3.12 -y
echo [$(date)]: "activating the env"
source ./env/Scripts/activate 
echo [$(date)]: "installing required packages"
pip install -r requirements.txt
echo [$(date)]: "END"