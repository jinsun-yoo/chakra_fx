# 2025-07-20
## Setup devcontainer env
For some reason, python ide within devcontainer is complaining about flakey errors
Turns out .devcontainer.json installs flake8 extension. Flake 8 is a linter, which is an alternative to Ruff linter.
However, we already have Ruff linter. Solution: Disable flake8 extension, install Ruff extension. edit devcontainer.json


Q1. How was previous instance able to get both forward and backward in one chakra graph?
Q2. How to get backward