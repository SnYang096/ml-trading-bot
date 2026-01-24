#!/usr/bin/env python3
"""
WSL2 WebSocket连接诊断脚本

检查WSL2网络配置、NAT转发、防火墙规则等，确定WebSocket连接问题的原因
"""
import subprocess
import sys
import os
import socket
import asyncio
import json
from pathlib import Path

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ websockets库未安装，部分测试无法进行")


def run_command(cmd, shell=False):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(),
            capture_output=True,
            text=True,
            shell=shell,
            timeout=10,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "命令超时", 1
    except Exception as e:
        return "", str(e), 1


def check_wsl2_environment():
    """检查WSL2环境"""
    print("=" * 80)
    print("1. WSL2环境检查")
    print("=" * 80)

    # 检查是否在WSL2中
    uname_out, _, _ = run_command("uname -a")
    print(f"系统信息: {uname_out}")

    if "microsoft" in uname_out.lower() or "WSL2" in uname_out:
        print("✅ 检测到WSL2环境")
    else:
        print("⚠️ 可能不是WSL2环境")

    # 检查内核版本
    kernel_out, _, _ = run_command("uname -r")
    print(f"内核版本: {kernel_out}")

    print()


def check_network_config():
    """检查网络配置"""
    print("=" * 80)
    print("2. 网络配置检查")
    print("=" * 80)

    # 检查IP地址
    ip_out, _, _ = run_command("hostname -I")
    print(f"WSL2 IP地址: {ip_out}")

    # 检查路由表
    print("\n路由表:")
    route_out, _, _ = run_command("ip route")
    print(route_out)

    # 检查DNS配置
    print("\nDNS配置:")
    if os.path.exists("/etc/resolv.conf"):
        with open("/etc/resolv.conf", "r") as f:
            print(f.read())
    else:
        print("⚠️ /etc/resolv.conf 不存在")

    # 检查默认网关
    print("\n默认网关:")
    gateway_out, _, _ = run_command("ip route | grep default")
    print(gateway_out if gateway_out else "未找到默认网关")

    print()


def check_firewall():
    """检查防火墙规则"""
    print("=" * 80)
    print("3. 防火墙检查")
    print("=" * 80)

    # 检查iptables规则
    print("iptables规则:")
    iptables_out, _, code = run_command("iptables -L -n -v 2>/dev/null", shell=True)
    if code == 0:
        print(iptables_out[:500] if len(iptables_out) > 500 else iptables_out)
    else:
        print("⚠️ 无法读取iptables规则（可能需要sudo）")

    # 检查ufw状态
    ufw_out, _, code = run_command("ufw status 2>/dev/null", shell=True)
    if code == 0:
        print(f"\nufw状态: {ufw_out}")
    else:
        print("\n⚠️ ufw未安装或无法访问")

    print()


def check_nat_forwarding():
    """检查NAT转发配置"""
    print("=" * 80)
    print("4. NAT转发检查")
    print("=" * 80)

    # 检查IP转发是否启用
    ip_forward, _, _ = run_command("cat /proc/sys/net/ipv4/ip_forward")
    print(f"IP转发状态: {'启用' if ip_forward == '1' else '禁用'}")

    # 检查NAT表
    nat_out, _, code = run_command("iptables -t nat -L -n -v 2>/dev/null", shell=True)
    if code == 0:
        print(f"\nNAT表规则:")
        print(nat_out[:500] if len(nat_out) > 500 else nat_out)
    else:
        print("\n⚠️ 无法读取NAT表（可能需要sudo）")

    print()


def test_tcp_connection(host, port):
    """测试TCP连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_network_connectivity():
    """测试网络连通性"""
    print("=" * 80)
    print("5. 网络连通性测试")
    print("=" * 80)

    test_targets = [
        ("fstream.binance.com", 443, "Binance主网HTTPS"),
        ("fstream.binance.com", 80, "Binance主网HTTP"),
        ("stream.binancefuture.com", 443, "Binance测试网HTTPS"),
        ("stream.binancefuture.com", 80, "Binance测试网HTTP"),
        ("8.8.8.8", 53, "Google DNS"),
        ("1.1.1.1", 53, "Cloudflare DNS"),
    ]

    for host, port, desc in test_targets:
        print(f"测试 {desc} ({host}:{port}): ", end="")
        if test_tcp_connection(host, port):
            print("✅ 连接成功")
        else:
            print("❌ 连接失败")

    print()


async def test_websocket_handshake(url, name):
    """测试WebSocket握手"""
    if not WEBSOCKETS_AVAILABLE:
        print(f"⚠️ {name}: websockets库未安装，跳过测试")
        return False

    print(f"测试 {name}:")
    print(f"  URL: {url}")
    try:
        async with websockets.connect(
            url,
            open_timeout=10,
            ping_interval=None,  # 禁用ping
        ) as ws:
            print(f"  ✅ WebSocket握手成功")
            # 尝试接收一条消息
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                print(f"  ✅ 收到消息: {str(msg)[:100]}")
                return True
            except asyncio.TimeoutError:
                print(f"  ⚠️ 握手成功但未收到消息（可能正常）")
                return True
    except asyncio.TimeoutError:
        print(f"  ❌ WebSocket握手超时")
        return False
    except Exception as e:
        print(f"  ❌ WebSocket连接失败: {type(e).__name__}: {e}")
        return False


async def test_websocket_connections():
    """测试WebSocket连接"""
    print("=" * 80)
    print("6. WebSocket连接测试")
    print("=" * 80)

    if not WEBSOCKETS_AVAILABLE:
        print("⚠️ websockets库未安装，跳过WebSocket测试")
        print("   安装: pip install websockets")
        return

    test_urls = [
        ("wss://fstream.binance.com/stream?streams=btcusdt@trade", "Binance主网组合流"),
        (
            "wss://stream.binancefuture.com/stream?streams=btcusdt@trade",
            "Binance测试网组合流",
        ),
    ]

    results = []
    for url, name in test_urls:
        result = await test_websocket_handshake(url, name)
        results.append((name, result))
        print()

    print("WebSocket测试总结:")
    for name, result in results:
        print(f"  {name}: {'✅ 成功' if result else '❌ 失败'}")

    print()


def check_proxy_settings():
    """检查代理设置"""
    print("=" * 80)
    print("7. 代理设置检查")
    print("=" * 80)

    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]

    print("环境变量:")
    has_proxy = False
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
            has_proxy = True

    if not has_proxy:
        print("  ✅ 未设置代理环境变量")

    # 检查git代理配置
    git_http_proxy, _, _ = run_command("git config --global http.proxy")
    git_https_proxy, _, _ = run_command("git config --global https.proxy")

    if git_http_proxy or git_https_proxy:
        print(f"\nGit代理配置:")
        if git_http_proxy:
            print(f"  http.proxy = {git_http_proxy}")
        if git_https_proxy:
            print(f"  https.proxy = {git_https_proxy}")
    else:
        print("\n✅ Git未配置代理")

    print()


def check_wsl2_specific_issues():
    """检查WSL2特定问题"""
    print("=" * 80)
    print("8. WSL2特定问题检查")
    print("=" * 80)

    # 检查.wslconfig文件（Windows端）
    print("WSL2配置:")
    print("  ⚠️ .wslconfig文件在Windows端，需要手动检查")
    print("  路径: C:\\Users\\<YourUser>\\.wslconfig")
    print("  建议检查:")
    print("    - [wsl2]")
    print("    - networkingMode=mirrored  # 镜像模式可能影响网络")
    print("    - dnsTunneling=true        # DNS隧道可能影响连接")

    # 检查Windows防火墙（通过WSL2）
    print("\nWindows防火墙:")
    print("  ⚠️ Windows防火墙在Windows端，需要手动检查")
    print("  建议:")
    print("    1. 打开Windows Defender防火墙")
    print("    2. 检查是否阻止了WSL2的网络连接")
    print("    3. 尝试临时关闭防火墙测试")

    print()


def generate_recommendations():
    """生成建议"""
    print("=" * 80)
    print("9. 诊断建议")
    print("=" * 80)

    recommendations = [
        "1. 如果WebSocket握手超时：",
        "   - 检查Windows防火墙是否阻止了WSL2的网络连接",
        "   - 尝试在Windows端添加防火墙规则允许WSL2",
        "   - 检查VPN是否支持WebSocket协议",
        "",
        "2. 如果NAT转发有问题：",
        "   - 检查WSL2的.wslconfig配置",
        "   - 尝试使用mirrored networking模式",
        "   - 检查Windows端的网络适配器设置",
        "",
        "3. 如果代理设置干扰：",
        "   - 临时取消HTTP_PROXY环境变量",
        "   - 检查VPN的代理设置",
        "   - 尝试直接连接（不使用代理）",
        "",
        "4. 如果问题持续：",
        "   - 考虑在服务器端运行（推荐）",
        "   - 使用Docker容器部署",
        "   - 使用云服务器（AWS EC2, 阿里云等）",
    ]

    for rec in recommendations:
        print(rec)

    print()


async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("WSL2 WebSocket连接诊断工具")
    print("=" * 80 + "\n")

    # 运行各项检查
    check_wsl2_environment()
    check_network_config()
    check_firewall()
    check_nat_forwarding()
    test_network_connectivity()
    await test_websocket_connections()
    check_proxy_settings()
    check_wsl2_specific_issues()
    generate_recommendations()

    print("=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n诊断被用户中断")
    except Exception as e:
        print(f"\n\n诊断过程中出错: {e}")
        import traceback

        traceback.print_exc()
