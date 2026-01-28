FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Build conda env
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Copy repo + install it (matches README: pip install -e .)
COPY . /app
RUN conda run -n facter-repro pip install -e .

# Auto-activate env for interactive shells
RUN CONDA_BASE="$(conda info --base)" && \
    echo ". ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate facter-repro" >> /etc/bash.bashrc

# Persistent environment, no experiments auto-start
CMD ["bash", "-lc", "sleep infinity"]