# Gunakan image resmi Python sebagai base image
FROM python:3.11-slim

# Set lingkungan kerja di dalam container
WORKDIR /app

# Copy requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file dari direktori lokal ke dalam container
COPY . .

# Set variabel lingkungan untuk memberitahu Flask untuk menjalankan aplikasi
ENV FLASK_APP=app.py

# Expose port yang digunakan oleh aplikasi Flask
EXPOSE 5003

# Perintah untuk menjalankan aplikasi
CMD ["python", "app.py"]