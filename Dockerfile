FROM python:3.8
WORKDIR /usr/local/src/sketch

RUN apt-get update && apt-get install wget ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

ARG DOWNLOAD_MODEL_PATH
RUN mkdir model && wget -qO model/model_final.pth "$DOWNLOAD_MODEL_PATH"

COPY main.py .
COPY recognition.py .

EXPOSE 5005
ENTRYPOINT ["python", "main.py"]