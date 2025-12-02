FROM hansenlovefiona017/lightgbm-runtime:v0.0.5

# Developer tools inside Dev Container (keeps parity with production image)
RUN pip3 install --no-cache-dir \
    black \
    flake8 \
    pylint \
    mypy \
    isort \
    ipython \
    notebook \
    jupyterlab

# Common CLI tools to make the devcontainer feel like a full Linux dev box
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        sudo \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        pkg-config \
        ssh \
        vim \
        unzip \
        htop \
        tmux \
        ripgrep \
        fd-find \
        fzf \
        fish \
    && rm -rf /var/lib/apt/lists/*

# Starship prompt (for nicer shell experience in bash & fish)
RUN curl -fsSL https://starship.rs/install.sh | bash -s -- -y && \
    echo 'eval \"$(starship init bash)\"' >> /etc/bash.bashrc && \
    mkdir -p /etc/fish && \
    echo 'eval \"$(starship init fish)\"' >> /etc/fish/config.fish

# Create yin user to match host user (UID 1000)
ARG USERNAME=yin
ARG USER_UID=1000
ARG USER_GID=1000
RUN if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        groupadd --gid "${USER_GID}" "${USERNAME}" && \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi && \
    mkdir -p /etc/sudoers.d && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    # Ensure workspace directory is accessible
    mkdir -p /workspace && \
    chown -R ${USERNAME}:${USERNAME} /workspace

# Marker for Makefile container detection
RUN touch /.devcontainer-env

WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]

