# Install Deep Learning Dependencies for Sequence Feature Extraction
# Supports: PyTorch, Mamba, FlashAttention

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Deep Learning Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running with administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Warning: Not running as administrator. Some installations may fail." -ForegroundColor Yellow
    Write-Host ""
}

# 1. Install PyTorch with CUDA support (RTX 3080)
Write-Host "[1/4] Installing PyTorch with CUDA support..." -ForegroundColor Green
Write-Host "      (This may take several minutes)" -ForegroundColor Gray

try {
    # PyTorch 2.1+ with CUDA 12.1 (recommended for RTX 3080)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Write-Host "      ✓ PyTorch installed" -ForegroundColor Green
} catch {
    Write-Host "      ✗ PyTorch installation failed: $_" -ForegroundColor Red
    Write-Host "      Trying alternative method..." -ForegroundColor Yellow
    pip install torch
}

Write-Host ""

# 2. Install Mamba (optional, highly recommended for efficiency)
Write-Host "[2/4] Installing Mamba (O(n) complexity SSM)..." -ForegroundColor Green

try {
    pip install mamba-ssm
    Write-Host "      ✓ Mamba installed" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Mamba installation failed: $_" -ForegroundColor Yellow
    Write-Host "      Mamba is optional. System will use Transformer instead." -ForegroundColor Gray
}

Write-Host ""

# 3. Install FlashAttention (optional, recommended for speedup)
Write-Host "[3/4] Installing FlashAttention (2-4x speedup)..." -ForegroundColor Green
Write-Host "      (This requires CUDA and may take time to compile)" -ForegroundColor Gray

try {
    pip install flash-attn --no-build-isolation
    Write-Host "      ✓ FlashAttention installed" -ForegroundColor Green
} catch {
    Write-Host "      ✗ FlashAttention installation failed: $_" -ForegroundColor Yellow
    Write-Host "      FlashAttention is optional. System will use standard Transformer." -ForegroundColor Gray
}

Write-Host ""

# 4. Install additional ML dependencies
Write-Host "[4/4] Installing additional ML dependencies..." -ForegroundColor Green

try {
    pip install scikit-learn pandas numpy scipy lightgbm
    Write-Host "      ✓ Additional dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Some dependencies failed: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test installations
Write-Host "Testing installations..." -ForegroundColor Cyan
Write-Host ""

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}: ', 'CUDA Available' if torch.cuda.is_available() else 'CPU Only')"

# Test Mamba
python -c "try: import mamba_ssm; print('Mamba: Available')
except: print('Mamba: Not available (optional)')"

# Test FlashAttention
python -c "try: import flash_attn; print('FlashAttention: Available')
except: print('FlashAttention: Not available (optional)')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python scripts/rolling/monthly_rolling_2025_with_feature_management.py" -ForegroundColor Gray
Write-Host "  2. Or use: make rolling-2025-advanced" -ForegroundColor Gray
Write-Host ""

