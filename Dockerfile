FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install pandas scikit-learn flask
CMD ["python", "app.py"]
