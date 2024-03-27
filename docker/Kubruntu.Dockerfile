# --- Image with *pre-installed* Kubric python package
# 
# docker run --rm --interactive \
#   --user $(id -u):$(id -g) \
#   --volume "$PWD:/kubric" \
#   --workdir "/kubric" \
#   kubricdockerhub/kubruntu:latest \
#   python3 examples/helloworld.py

FROM kubricdockerhub/blender:latest

WORKDIR /kubric

# --- copy requirements in workdir
COPY requirements.txt .
COPY requirements_full.txt .

# --- install python dependencies
RUN pip install --upgrade pip wheel
RUN pip install --upgrade --force-reinstall -r requirements.txt
RUN pip install --upgrade --force-reinstall -r requirements_full.txt

# --- cleanup
RUN rm -f requirements.txt
RUN rm -f requirements_full.txt

# --- Silences tensorflow
ENV TF_CPP_MIN_LOG_LEVEL="3"

# --- Install Kubric
RUN git clone https://github.com/jeongyw12382/kubric_gpu.git
RUN cd kubric_gpu && pip3 install -e .
