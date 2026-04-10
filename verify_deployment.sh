#!/bin/bash
# Quick deployment verification script
# Run this to validate the v3.0 deployment

set -e

echo "🚀 Brain-Connectivity-GCN v3.0 Deployment Verification"
echo "========================================================"
echo ""

# Check Python version
echo "✓ Python version:"
python --version
echo ""

# Check key packages
echo "✓ Checking dependencies..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ⚠️  PyTorch not installed"
python -c "import pytorch_lightning; print(f'  PyTorch Lightning: {pytorch_lightning.__version__}')" 2>/dev/null || echo "  ⚠️  PyTorch Lightning not installed"
python -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  ⚠️  scikit-learn not installed"
echo ""

# Run test suite
echo "✓ Running test suite..."
python -m pytest tests/ -q --tb=no
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✅ ALL TESTS PASSING - Deployment Ready!"
    echo ""
    echo "Next steps:"
    echo "  1. python -m brain_gcn.main --help  (view CLI options)"
    echo "  2. Review DEPLOYMENT.md for detailed changes"
    echo "  3. python -m brain_gcn.main --max_epochs 1 --max_windows 5 (smoke test)"
else
    echo ""
    echo "❌ Some tests failed - Review output above"
    exit 1
fi
