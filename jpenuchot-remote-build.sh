#!/bin/bash

HOST=$1
REMOTE_DIR=$2

echo
echo ==========================================
echo Host: $1
echo Remote dir: $2
echo ==========================================
echo

rsync -r --exclude-from=.git/info/exclude --exclude-from=.gitignore . $HOST:$REMOTE_DIR
ssh $HOST "cd $REMOTE_DIR && ./jpenuchot-builder.sh"
