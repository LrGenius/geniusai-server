@echo off

echo Building geniusai-server...
md dist

REM Check for conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo conda could not be found, please install it first.
    exit /b 1
)

REM Check if the environment exists
conda env list | findstr /C:"geniusai_server" >nul
if %errorlevel% neq 0 (
    echo Conda environment 'geniusai_server' not found, creating it...
    call conda env create -f environment.yml
    call conda activate geniusai_server
)

call conda activate geniusai_server

REM Check if pyinstaller is installed
where pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
    echo pyinstaller could not be found, please install it in the geniusai_server environment.
    exit /b 1
)

REM Set environment variable to fix OpenMP library conflict during build
set KMP_DUPLICATE_LIB_OK=TRUE

pyinstaller geniusai_server.spec --noconfirm

call conda deactivate

echo Build complete. The executable can be found in the dist/ directory.
