#!/bin/bash

# Function to safely change directory
cd_safe() {
    if ! cd "$1"; then
        echo "Error: Failed to change directory to $1"
        exit 1
    fi
}

ensure_network() {
    if ! docker network inspect main-network &>/dev/null; then
        echo "Creating network..."
        docker network create main-network
    fi
}



up_backend() {
    echo "Starting backend service..."
    ensure_network
    cd_safe backend
    docker compose up -d
    echo "Backend service is up."
    cd_safe ..
}

up_vllm() {
    echo "Starting VLLM service..."
    ensure_network
    cd_safe vllm_api
    docker compose up -d
    echo "VLLM service is up."
    cd_safe ..
}

up_monitor() {
    echo "Starting monitoring services..."
    ensure_network
    cd_safe monitor
    docker compose up -d
    echo "Monitoring services are up."
    cd_safe ..
}



build_backend() {
    echo "Building backend service..."
    cd_safe backend
    docker compose build
    echo "Backend service is built."
    cd_safe ..
}

build_vllm() {
    echo "Building VLLM service..."
    cd_safe vllm_api
    docker compose build
    echo "VLLM service is built."
    cd_safe ..
}

build_monitor() {
    echo "Building monitoring services..."
    cd_safe monitor
    docker compose build
    echo "Monitoring services are built."
    cd_safe ..
}



down_backend() {
    echo "Stopping backend service..."
    cd_safe backend
    docker compose down
    echo "Backend service is stopped."
    cd_safe ..
}

down_vllm() {
    echo "Stopping VLLM service..."
    cd_safe vllm_api
    docker compose down
    echo "VLLM service is stopped."
    cd_safe ..
}

down_monitor() {
    echo "Stopping monitoring services..."
    cd_safe monitor
    docker compose down
    echo "Monitoring services are stopped."
    cd_safe ..
}

up_services() {
    up_monitor
    up_vllm
    up_backend
    echo "All services are up and running."
}

build_services() {
    build_monitor
    build_vllm
    build_backend
    echo "All services are built."
}

build_up_services() {
    echo "Building and starting all services..."
    build_services
    up_services
}

down_services() {
    down_backend
    down_vllm
    down_monitor
    echo "All services have been stopped."
}

restart_services() {
    echo "Restarting all services..."
    down_services
    up_services
}

status_services() {
    echo "Checking status of all services..."
    
    echo "=== Backend Services ==="
    cd_safe backend
    docker compose ps
    cd_safe ..
    
    echo "=== VLLM Services ==="
    cd_safe vllm_api
    docker compose ps
    cd_safe ..
    
    echo "=== Monitoring Services ==="
    cd_safe monitor
    docker compose ps
    cd_safe ..
}

help() {
    echo "Usage: $0 {up|up-all|build|build-up|down|restart|status|help}"
    echo "Commands:"
    echo "  up          Start all services on one machine"
    echo "  up-all      Alias for up"
    echo "  build       Build all services"
    echo "  build-up    Build all services and then start them"
    echo "  down        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  help        Show this help message"
}

# Main script logic
if [ $# -eq 0 ]; then
    help
    exit 1
fi

case "$1" in
    up)
        up_services
        ;;
    up-all)
        up_services
        ;;
    build)
        build_services
        ;;
    build-up)
        build_up_services
        ;;
    down)
        down_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    help)
        help
        ;;
    *)
        echo "Invalid command: $1"
        help
        exit 1
        ;;
esac