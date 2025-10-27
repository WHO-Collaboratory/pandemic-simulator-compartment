FROM public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Implicitly copies batch helpers when in deploy-ecr-tag action
COPY . .
CMD ["python", "main.py"]