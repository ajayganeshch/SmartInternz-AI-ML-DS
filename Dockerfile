FROM python:3.7-slim

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip
RUN pip install --upgrade --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

CMD ["python", "app.py"]
