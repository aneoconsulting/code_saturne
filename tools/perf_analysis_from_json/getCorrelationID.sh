#!/bin/bash

FILE=$1
KERNELID=$2
STREAMID=$3

if [ -z "$FILE" ] || [ -z "$KERNELID" ]; then
    echo "Usage: $0 <file> <kernel_id> [stream_id]" >&2
    exit 1
fi

if [ -z "$STREAMID" ]; then
    # Only filter by kernel ID
    rg -F "\"kernel\":{\"demangledName\":\"${KERNELID}\"" "$FILE" | \
    awk -F',' '{print $7}' | uniq
else
    # Filter by kernel ID, then by stream ID (first match only)
    rg -F "\"kernel\":{\"demangledName\":\"${KERNELID}\"" "$FILE" | \
    rg -F -m 1 "\"streamId\":\"${STREAMID}\"" | \
    awk -F',' '{print $4}'
fi

