FROM prom/prometheus:latest

# Chuyển sang quyền root để thao tác file
USER root

COPY . /etc/prometheus/

# Cấp quyền thực thi và bàn giao quyền sở hữu thư mục cho user nobody
RUN chmod +x /etc/prometheus/entrypoint.sh && \
    chown -R nobody:nobody /etc/prometheus

# Trả lại quyền chạy mặc định cho user nobody của Prometheus
USER nobody

ENTRYPOINT ["/bin/sh", "/etc/prometheus/entrypoint.sh"]