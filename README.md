# Environment #

* Ruby 3.1.2
* Jekyll 4.2.2
* WEBRick 1.7.0

# Build #

Running Jekyll:

```
$ rm -rf _site/
$ jekyll serve
```

# Deployment #

Copying the contents of the newly generated `_site` directory to the GitHub deployment repository:

```
$ rm -rf ../openscoring.github.io/*
$ cp -r _site/* ../openscoring.github.io/
```