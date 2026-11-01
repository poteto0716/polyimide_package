#!/bin/sh
# Decrypt the training data into this folder. gpg asks for the password.
set -e
cd "$(dirname "$0")"
gpg --decrypt training_data.tar.gz.gpg | tar xzf -
echo "Decrypted. Do not redistribute the extracted files."
