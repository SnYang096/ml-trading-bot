FROM hansenlovefiona017/lightgbm-runtime:v0.0.7

# Developer tools inside Dev Container (keeps parity with production image)
# Increase timeout and add retries to handle network issues
RUN pip3 install --no-cache-dir --default-timeout=300 --retries 5 \
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
RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y -b /usr/local/bin && \
    starship --version && \
    echo 'eval "$(starship init bash)"' >> /etc/bash.bashrc && \
    mkdir -p /etc/fish && \
    echo 'eval "$(starship init fish)"' >> /etc/fish/config.fish

# Create dev user to match host user (UID 1000), handle existing UID/GID gracefully
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN /bin/bash -lc '\
    set -eu; \
    EXISTING_GROUP=$(getent group "${USER_GID}" | cut -d: -f1 || true); \
    if [ -z "${EXISTING_GROUP}" ]; then \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    else \
    echo "Group with GID ${USER_GID} already exists: ${EXISTING_GROUP}"; \
    fi; \
    EXISTING_UID_USER=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
    if [ -n "${EXISTING_UID_USER}" ]; then \
    echo "UID ${USER_UID} already exists (user: ${EXISTING_UID_USER}), creating ${USERNAME} with same UID/GID (non-unique)"; \
    useradd -o --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" || true; \
    elif ! id -u "${USERNAME}" >/dev/null 2>&1; then \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    else \
    echo "User ${USERNAME} already exists"; \
    fi; \
    mkdir -p /etc/sudoers.d; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}; \
    mkdir -p /workspace; \
    chown -R ${USER_UID}:${USER_GID} /workspace \
    '

# Marker for Makefile container detection
RUN touch /.devcontainer-env

WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]

