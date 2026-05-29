FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY handler.py .
COPY inference.py .

CMD ["python", "-u", "handler.py"]