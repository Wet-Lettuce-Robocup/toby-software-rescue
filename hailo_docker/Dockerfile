# Use Ubuntu 22.04 as the base image (x86_64)
FROM ubuntu:22.04

# Set non-interactive mode and disable .pyc files / buffering
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    python3-tk \
    build-essential \
    liblapack-dev \
    libatlas-base-dev \
    gcc \
    curl \
    git \
    graphviz \
    libgraphviz-dev \
    libgl1-mesa-glx \
    pkg-config \
    lsb-release \
    bsdmainutils \
    # Gstreamer is probably not needed. Consider removing it.
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 && \
    rm -rf /var/lib/apt/lists/*


# Set working directory for our app
WORKDIR /app

# Create a virtual environment in /venv
# The venv is required, otherwise the Hailo Dataflow Compiler will not work properly
RUN python3.10 -m venv /venv && \
    ls -l /venv

# Set environment variables so that the venv is used
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:/opt/hailo/usr/bin:$PATH"

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy the Hailo Dataflow Compiler wheel into the image and install it in the venv
COPY hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl

# Install the pre-built HailoRT deb package (runtime only)
# Note dkpg will not work here, because the post-install scripts inside the .deb cannot run during Docker build.
# We are putting it here in a non-standard location
COPY hailort_5.3.0_amd64.deb /tmp/
RUN mkdir -p /opt/hailo && dpkg-deb -x /tmp/hailort_5.3.0_amd64.deb /opt/hailo

# Create symlinks for hailortcli and the library
RUN ln -s /opt/hailo/usr/bin/hailortcli /usr/bin/hailortcli && \
    ln -s /opt/hailo/usr/lib/libhailort.so.5.3.0 /usr/lib/libhailort.so

# Switch to root temporarily to set up permissions for the SDK client directory
USER root
RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    chown -R appuser /app && \
    # Not sure if this next line is still needed. Maybe can be removed.
    mkdir -p /usr/local/lib/python3.10/sdk_client && \
    chmod -R 777 /usr/local/lib/python3.10/sdk_client && \
    chown -R appuser /venv

# Copy requirements file and install Python dependencies in the venv
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy project files into the container
COPY src/ ./src/
COPY models/ ./models/
COPY calibration_images/ ./calibration_images/

USER appuser

# Set environment variables so the Hailo SDK can find its shared libraries and binaries
ENV LD_LIBRARY_PATH=/opt/hailo/usr/lib
ENV PATH=$PATH:/opt/hailo/usr/bin

# Use forward slashes in CMD; the virtualenv is already activated because PATH is set accordingly.
CMD ["python", "src/main.py"]