#!/usr/bin/env bash

echo "🔧 升級 pip setuptools wheel 中..."
pip install --upgrade pip setuptools wheel

echo "📦 安裝 requirements.txt..."
pip install -r requirements.txt
