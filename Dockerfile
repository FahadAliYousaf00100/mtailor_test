FROM python:3.12-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8192

CMD ["uvicorn", "testserver:app", "--host", "0.0.0.0", "--port", "8192"]
