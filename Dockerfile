FROM jjanzic/docker-python3-opencv

WORKDIR /app

COPY . .
RUN sed -i '/opencv-python/d' ./requirements.txt \
 && pip install --no-cache-dir -r requirements.txt