---
layout: post
title: Image generate
date: 2025-02-13
description: this is super image
tags: formatting images
categories: sample-posts
images:
  lightbox2: true
---

{% assign image_files = site.static_files | where: "image", true %}
{% for myimage in image_files %}
  {{ myimage.path }}
{% endfor %}

The images in this post are all zoomable, arranged into different mini-galleries using different libraries.

## [Lightbox2](/assets/images/ComfyUI_00006_.png)

<a href="/assets/images/ComfyUI_00006_.png" data-lightbox="roadtrip"><img src="/assets/images/ComfyUI_00006_.png" /></a>

---
