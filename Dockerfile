FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends     git build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /workspace

CMD ["bash"]