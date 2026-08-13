#!/bin/bash
# Setup script for project-robotics with Python virtual environment
# Este script configura um venv e instala todas as dependências necessárias

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/robotics"

echo "================================================"
echo "Configurando ambiente virtual para project-robotics"
echo "================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 não encontrado. Por favor, instale Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "OK - $PYTHON_VERSION encontrado"
echo ""

# Remove old venv if it exists and is broken
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Removendo virtual environment incompleto..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Criando virtual environment em $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Erro ao criar virtual environment"
        exit 1
    fi
    echo "OK - Virtual environment criado"
else
    echo "OK - Virtual environment ja existe"
fi

echo ""
echo "Atualizando pip, setuptools e wheel..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "Erro ao atualizar pip"
    exit 1
fi
echo "OK - Ferramentas base atualizadas"

echo ""
echo "Instalando dependencias do projeto..."
"$VENV_DIR/bin/pip" install -r "$PROJECT_ROOT/docs/requirements.txt"
if [ $? -ne 0 ]; then
    echo "Erro ao instalar dependências"
    exit 1
fi
echo "OK - Todas as dependencias instaladas"

echo ""
echo "================================================"
echo "Ambiente configurado com sucesso!"
echo "================================================"
echo ""
echo "Para usar o ambiente virtual, execute:"
echo "  source robotics/bin/activate"
echo ""
echo "Para desativar em qualquer momento:"
echo "  deactivate"
echo ""
