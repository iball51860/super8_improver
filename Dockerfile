FROM jjanzic/docker-python3-opencv

COPY requirements.txt ./
RUN sed -i '/opencv-python/d' ./requirements.txt \
 && pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/project

CMD [ "python", "./main.py" ]