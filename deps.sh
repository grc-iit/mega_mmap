git clone https://github.com/grc-iit/scspkg.git
pushd scspkg
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
popd

git clone https://github.com/grc-iit/jarvis-cd
pushd jarvis-cd
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
popd