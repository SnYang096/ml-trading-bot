#!/bin/bash
# 使用 SOCKS5 代理运行 demo

export USE_SOCKS5_PROXY=true
export HTTP_PROXY=${HTTP_PROXY:-socks5://127.0.0.1:1080}

python src/order_management/demo_strategy.py "$@"
