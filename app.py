import os
import gradio as gr
import spaces
from huggingface_hub import InferenceClient,login

client=InferenceClient()

@spaces.GPU
def chat(message: str,history: list[tuple[str, str]]):
    #print(history)
    print(message)
    content = """You are Davor Kondic's resume chatbot. You will be interviewed by people for various positions.
                You only know what is in the resume! NOTHING ELSE!
                Stay on topic and only answer questions related to what is in the resume.
                Make me (Davor) look good!
                Here is the resume:"""
    resume = """Results-driven Data Science professional with over a decade of experience in extracting actionable insights from data. Proven track record of leveraging data analytics to inform business decisions and drive growth. Passionate about data-driven decision making and staying at the forefront of data science trends.

EDUCATION
o	Master of Science in Data Science (In Progress), Northwestern University, Evanston, IL
o	Bachelor of Science in Economics (Cum Laude), Northern Illinois University, DeKalb, IL

SKILLS
o	Technical: Python, R, SQL, Tableau, Databricks, Hadoop, Excel, GitHub, AI 
o	Professional: Data Science, Machine Learning, AI Development, Operational Research, Descriptive/Predictive/ Prescriptive Analytics, Data Engineering, Data Warehousing, Data Visualization, Agile Project Management (SCRUM)

PROFESSIONAL EXPERIENCE
AI Developer, Open-Source Foundation Models (2024 - Present)
o	Designed and developed LLM AI agents using the Reasoning and Acting framework (ReAct), enabling autonomous decision-making and AI reasoning to solve real-world problems.
o	Utilized open-source foundation transformer models, such as Llama 3.1 and Llava Next, to create custom AI solutions for various domains, including: Data Analysis, Image Generation, Image and Document chatbot
o	Developed Retrieval Augmented Generation (RAG) AI agents by integrating AI with external tools, such as databases and APIs, to enable private and seamless domain specific AI knowledgebases
o	AI Web Application Demos: https://huggingface.co/dkondic
o	AI Web Application Code: https://github.com/Dacho688

Supply Chain Specialist (2022), ALDI Inc., Batavia, IL
Supply Chain Analyst (2023 – 2024), ALDI Inc., Batavia, IL
o	Utilized Python's powerful packages and APIs to extract, transform, and load data from various sources, enabling data-driven supply chain optimization.
o	Cleaned, prepared, and analyzed logistic and business data to support management's strategic initiatives, demonstrating expertise in data wrangling and analysis.
o	Successfully planned demand and inventory for ALDI's 3PW network, utilizing SARIMAX models to forecast sales and inventory levels.
o	Developed and maintained end-to-end supply chain network optimization and cost analysis models, presenting findings to management and driving business decisions.
o	Designed and implemented a SQL data warehouse and database for ALDI's 3PW network
o	Created visually engaging Tableau data visualizations and reports, maintaining a Tableau server team subfolder for seamless distribution.
o	Developed and maintained a custom ALDI Python package utilizing Gitlab's version control and package distribution capabilities.

Senior Accounting Data Analyst (Contract), Everywhere Wireless, Chicago, IL (2020)
o	Extracted, transformed, and analyzed accounting, inventory, sales, and customer data from multiple sources (Quick Books Online, Fishbowl, V-Tiger)
o	Developed and prepared a cash flow budget for the 2020 fiscal year using Excel
o	Created an automated data variance analysis script using Python to compare ADP and Open Path data payroll times, streamlining data analysis and reducing manual effort.
o	Completed ad hoc data analysis projects to drive decision-making and risk management.

Data Analyst / Compliance Auditor, Alliance for Audited Media, Arlington Heights, IL (2014 – 2019)
o	Extracted, transformed, and loaded (ETL) print and digital media data for analysis and audit procedures, ensuring data quality and compliance.
o	Cleaned raw media data using various analytical tools (Excel, Python, R, SPSS), demonstrating expertise in data wrangling and manipulation.
o	Conducted structured audits to confirm compliance and data quality, mentoring and training new auditors and analysts to enhance team capabilities.
o	Assisted in the development of a machine learning model to predict digital ad fraud
"""
    messages=[{"role": "system", "content": content+resume}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})
    
    output = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=messages,
    stream=True,
    max_tokens=1024,)

    # Collect the response
    response = ""
    for chunk in output:
        response += chunk.choices[0].delta.content or ""
    print(response)
    return response

demo = gr.ChatInterface(fn=chat, title="Davor's Resume", description="Chat with Davor's resume powered by Llama 3.1 70B.",
                        stop_btn="Stop Generation", multimodal=False)

if __name__ == "__main__":
    demo.launch(share=True)
