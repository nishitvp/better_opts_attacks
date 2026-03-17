#!/usr/bin/env bash

# Project root = current directory when sourced from repo root
export PROJECT_ROOT="$(pwd)"

# Generic XDG locations for tools that honor them
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"

# uv
export UV_INSTALL_DIR="$PROJECT_ROOT/.local/bin"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.local/share/uv/python"
export UV_TOOL_DIR="$PROJECT_ROOT/.local/share/uv/tools"
export UV_TOOL_BIN_DIR="$PROJECT_ROOT/.local/bin"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

# Optional: if you want uv-managed python shims here too
export UV_PYTHON_BIN_DIR="$PROJECT_ROOT/.local/bin"

# pip fallback cache
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"

# Hugging Face / Transformers
export HF_HOME="$PROJECT_ROOT/.hf"
export HF_HUB_CACHE="$PROJECT_ROOT/.hf/hub"

# PyTorch
export TORCH_HOME="$PROJECT_ROOT/.torch"

# Put local bin first
export PATH="$PROJECT_ROOT/.local/bin:$PATH"