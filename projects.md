---
layout: page
title: Projects
permalink: /projects/
---

<div class="projects-grid">
  {% assign sorted_projects = site.projects | sort: "order" %}
  {% for project in sorted_projects %}
  <div class="project-card">
    {% if project.image %}
    <div class="project-card-img">
      <img src="{{ project.image }}" alt="{{ project.title }}">
    </div>
    {% endif %}
    <div class="project-card-body">
      <h2 class="project-title">{{ project.title }}</h2>
      <p class="project-tags">{{ project.tags }}</p>
      <p class="project-tldr"><strong>TL;DR</strong> {{ project.tldr }}</p>
      <p class="project-desc">{{ project.description }}</p>
      <div class="project-links">
        {% if project.code and project.code != "" %}
        <a href="{{ project.code }}" target="_blank" class="project-btn">Code</a>
        {% endif %}
        {% if project.paper and project.paper != "" %}
        <a href="{{ project.paper }}" target="_blank" class="project-btn">Paper</a>
        {% endif %}
        {% if project.page and project.page != "" %}
        <a href="{{ project.page }}" target="_blank" class="project-btn">Project Page</a>
        {% endif %}
      </div>
    </div>
  </div>
  {% endfor %}
</div>
