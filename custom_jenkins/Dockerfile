# =========================
# Base image
# =========================
FROM jenkins/jenkins:lts


# =========================
# Switch to root
# =========================
USER root


# =========================
# Install Docker (modern & compatible)
# =========================
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo \
      "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/debian \
      bookworm stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# =========================
# Docker permissions
# =========================
RUN groupadd -f docker && \
    usermod -aG docker jenkins


# =========================
# Docker volume (DinD)
# =========================
VOLUME /var/lib/docker


# =========================
# Back to Jenkins user
# =========================
USER jenkins
