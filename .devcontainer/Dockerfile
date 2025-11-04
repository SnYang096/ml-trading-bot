FROM hansenlovefiona017/lightgbm-runtime:v0.0.3

# Developer tools inside Dev Container (keeps parity with production image)
RUN pip3 install --no-cache-dir \
    black \
    flake8 \
    pylint \
    mypy \
    notebook

# Marker for Makefile container detection
RUN touch /.devcontainer-env

WORKDIR /workspace


