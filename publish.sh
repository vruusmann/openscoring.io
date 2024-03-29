#!/bin/sh

export PUBLISH_DIR=../openscoring.github.io/

rm -rf _site/
JEKYLL_ENV=production bundle exec jekyll build

tidy -xml -utf8 -imq _site/sitemap.xml

# Read CNAME into memory
cname=`cat $PUBLISH_DIR/CNAME`
#echo $cname

rm -rf $PUBLISH_DIR/*
cp -r _site/* $PUBLISH_DIR

# Restore CNAME from memory
echo -n $cname > $PUBLISH_DIR/CNAME
