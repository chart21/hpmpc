#!/bin/sh
rm -rf *.pdf
if ls ../../../logs/node* 1> /dev/null 2>&1; then
    rm -rf ../../../logs/node*
fi
echo "Removed existing experiment data and plots"
