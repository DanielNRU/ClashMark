FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

RUN mkdir -p /app/model

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "web.routes:app"] 