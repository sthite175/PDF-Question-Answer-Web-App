FROM python:3.10

COPY . /home
WORKDIR /home

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD  ["python", "app.py"]