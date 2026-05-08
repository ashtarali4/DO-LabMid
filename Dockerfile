FROM python:3.9-slim
WORKDIR /app
COPY . .
# Add fastapi and uvicorn to the install list
RUN pip install pandas scikit-learn fastapi uvicorn numpy
# FastAPI usually runs via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
