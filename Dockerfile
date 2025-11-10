# 1. Use the official PyTorch 2.4.1 image with CUDA 12.1
# This image already includes Python, CUDA, CUDNN, and PyTorch
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# 2. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# 3. Set the working directory
WORKDIR /app

# 4. Copy and install *only* the extra requirements
# We use the requirements.txt file that has torch/torchvision removed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application's source code
COPY . .

# 6. Set default command to a bash shell
# This allows you to run the container interactively
CMD ["bash"]