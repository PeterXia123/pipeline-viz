FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
Maintainer Liam Pieri <pieri.liam@gmail.com>

COPY ./requirements.txt /usr/

RUN pip install --no-cache-dir -r /usr/requirements.txt

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]

