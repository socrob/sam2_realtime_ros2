all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build the Docker image"
	@echo "   2. make clean              - remove the Docker image"
	@echo ""

# Build the Docker image
build:
	@docker build --tag ros2_template -f ./Dockerfile ..

# Remove the Docker image
clean:
	@docker rmi -f ros2_template

# Build with Docker Compose
compose-build:
	docker compose -f docker-compose.yml build

# Run with Docker Compose
compose-up:
	xhost +local:docker
	docker compose -f docker-compose.yml run --rm ros2_template

# Clean image
compose-clean:
	docker rmi -f ros2_template

