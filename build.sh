#!/usr/bin/bash

pyinstaller src/main.py --noconfirm --clean --onefile --name cellua \
        --add-data VERSION:.
