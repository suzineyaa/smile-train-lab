#!/bin/sh
set -eu
mkdir -p /data
chown appuser:appuser /data
exec gosu appuser "$@"
