#!/bin/bash
# Deploy the pi_server to the Raspberry Pi.
# The server will be placed at ~/linetrace-server/ on the Pi.

set -e

PI_HOST="${1:-robo@roboberry.local}"
PI_DIR="linetrace-server"

echo "Deploying pi_server to ${PI_HOST}:~/${PI_DIR}/ ..."

rsync -avz --delete \
    pi_server/ \
    "${PI_HOST}:${PI_DIR}/"

echo ""
echo "Done. To run the server on the Pi:"
echo "  ssh ${PI_HOST}"
echo "  cd ${PI_DIR} && python3 server.py"
