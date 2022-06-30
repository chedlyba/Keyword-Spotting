# Pull tensorflow image with py3
FROM tensorflow/tensorflow:latest

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to working dirr
ADD . /app 

# Install audio libraries 
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg && pip install --upgrade pip -I && pip install --upgrade setuptools -I 

# Install uwsgi 
RUN apt install python3.8 python3.8-dev python3-distutils uwsgi uwsgi-src uuid-dev libcap-dev libpcre3-dev python3-pip python3.8-venv -y

# Prepare virtual environment
#RUN python3 -m venv /app 

# Install python packages
RUN pip install -r requirements.txt

# Start venv and start uWSGI using config file
CMD  ["uwsgi", "app.ini"]

