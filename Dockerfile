# Slim Python base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to keep stdout/stderr unbuffered for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install critical system-level dependencies required for mapping and modeling
# - build-essential: For compiling PyMC/PyTensor C-code
# - libgeos-dev, libproj-dev, proj-data: The C-libraries required by Cartopy/Shapely
# - libnetcdf-dev, libhdf5-dev: Required for saving PyMC traces to .nc files
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libproj-dev \
    proj-data \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Upgrade pip and install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the container
COPY . /app/

# Expose port 8888 so we can access Jupyter Notebooks from the browser
EXPOSE 8888

# Default command: Launch Jupyter Lab securely but accessibly for local development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=globe2026"]