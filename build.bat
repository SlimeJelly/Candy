@echo off
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administer permission required.
    pause > nul
    exit
)
set "origin_folder=%~dp0"
cd /D "%origin_folder%"
set /p name="enter the build name (press 'ENTER' to make auto): "
if "%name%"=="" set name=%date%

set "folder=build/%name%"
if not exist "%folder%" goto nextFolder-End

set n=1
:nextFolder
if exist "%folder%" (
    set /a n+=1
    set "folder=build/%name%-%n%"
    goto nextFolder
)
:nextFolder-End

mkdir "%folder%"
cd /D "%folder%"

pyinstaller "%origin_folder%/Candy.py"

rmdir /s "build/"
del Candy.spec

for /f "delims=" %%i in ('dir /b /s /a-d "dist\Candy\*"') do move "%%i" .
rmdir /s "dist/"