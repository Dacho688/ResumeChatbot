---
title: Resume Chatbot 
emoji: 🗎
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.22.0
app_file: app.py
pinned: false
short_description: Davor's resume chatbot powered by Llama-3.3-70B-Instruct
---

# Resume Chatbot

This project aims to illustrate AI's contextual awareness with private and/or domain specific AI knowledgebases (ie data not used durring model training). We utilize Meta's Llama-3.3-70B-Instruct open source large language model and feed it with external data, in this case my resume. We then build a Python based Gradio web app and deploy it to Huggingface Spaces. A github workflow is used to push any main branch commits to the Huggingface space automatically.

<a href="https://dkondic-resumechatbot.hf.space/">Try APP</a>
