FROM python:latest

WORKDIR /app

COPY app/ .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main.py"]
