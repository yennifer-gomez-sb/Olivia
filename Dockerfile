FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# sudo docker build -t wapp . --no-cache
# sudo docker run --rm -p 8080:8080 --name wapp-con wapp
# docker save -o myapp.tar my-app
