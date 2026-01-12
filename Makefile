# Makefile for Diffusion Boltzmann Sampler
#
# Usage:
#   make help      - Show this help
#   make install   - Install all dependencies
#   make dev       - Run both backend and frontend
#   make test      - Run all tests

.PHONY: help install dev backend frontend test lint clean

# Default target
help:
	@echo "Diffusion Boltzmann Sampler"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install all dependencies"
	@echo "  make dev         Run backend and frontend (use tmux or two terminals)"
	@echo "  make backend     Run backend server"
	@echo "  make frontend    Run frontend dev server"
	@echo "  make test        Run all tests"
	@echo "  make test-backend  Run backend tests only"
	@echo "  make test-frontend Run frontend tests only"
	@echo "  make lint        Run linting"
	@echo "  make format      Format code"
	@echo "  make clean       Clean build artifacts"
	@echo ""

# Installation
install: install-backend install-frontend

install-backend:
	@echo "Installing backend dependencies..."
	pip install -r requirements.txt

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Development servers
dev:
	@echo "Run 'make backend' and 'make frontend' in separate terminals"

backend:
	@echo "Starting backend server..."
	cd backend && uvicorn api.main:app --reload --port 8000

frontend:
	@echo "Starting frontend server..."
	cd frontend && npm run dev

# Testing
test: test-backend test-frontend

test-backend:
	@echo "Running backend tests..."
	pytest backend/tests/ -v

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm run test:run

# Linting
lint: lint-backend lint-frontend

lint-backend:
	@echo "Linting backend..."
	ruff check backend/
	mypy backend/ --ignore-missing-imports

lint-frontend:
	@echo "Linting frontend..."
	cd frontend && npm run lint

# Formatting
format: format-backend

format-backend:
	@echo "Formatting backend..."
	black backend/
	isort backend/

# Build
build-frontend:
	@echo "Building frontend..."
	cd frontend && npm run build

# Clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf backend/__pycache__
	rm -rf backend/**/__pycache__
	rm -rf .pytest_cache
	rm -rf frontend/dist
	rm -rf frontend/node_modules/.cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker (future)
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up
