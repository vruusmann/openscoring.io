#!/bin/sh

export PUBLISH_DIR=../openscoring.github.io/

rm -rf _site/
bundle exec jekyll build

# Read CNAME into memory
cname=`cat $PUBLISH_DIR/CNAME`
#echo $cname

rm -rf $PUBLISH_DIR/*
cp -r _site/* $PUBLISH_DIR

# Restore CNAME from memory
echo -n $cname > $PUBLISH_DIR/CNAME
