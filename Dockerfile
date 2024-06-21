FROM python:latest

WORKDIR /docker

COPY main.py .
COPY requirements.txt .
COPY lending_club_mlp_binary_classifier.pkl .

# RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main.py"]
