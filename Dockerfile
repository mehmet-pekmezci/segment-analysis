FROM python:3.8-slim

RUN mkdir -p /app/output

COPY ./input /app/input
COPY ./segmentation-analysis /app/segmentation-analysis
COPY ./requirements.txt /

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/segmentation-analysis

CMD ["python", "two-segment-train.py"]

