# Environment #

* Ruby 3.1.2
* Jekyll 4.2.2
* WEBRick 1.7.0

# Initialization #

Running Bundler:

```
$ bundle init
$ bundle add jekyll
$ bundle add webrick
```

# Build #

Running Jekyll:

```
$ rm -rf _site/
$ bundle exec jekyll build
```

# Deployment #

Publish to GitHub pages directory:

```
$ sh publish.sh
```
