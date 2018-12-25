FROM python:3.7.1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/project

CMD [ "python", "./main.py" ]