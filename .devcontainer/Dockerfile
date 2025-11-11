FROM hansenlovefiona017/lightgbm-runtime:v0.0.5

# Developer tools inside Dev Container (keeps parity with production image)
RUN pip3 install --no-cache-dir \
    black \
    flake8 \
    pylint \
    mypy \
    notebook

RUN apt-get update && \
    apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/*

# Create vscode user expected by devcontainer.json
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        groupadd --gid "${USER_GID}" "${USERNAME}" && \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home "${USERNAME}"; \
    fi && \
    mkdir -p /etc/sudoers.d && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Marker for Makefile container detection
RUN touch /.devcontainer-env

WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]

