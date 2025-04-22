import os
import gradio as gr
import spaces
from huggingface_hub import InferenceClient,login,whoami

if not whoami(False):
    login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    
client=InferenceClient()

@spaces.GPU
def chat(message: str,history: list[tuple[str, str]]):
    print("User Message: ",message,"\n")
    content = """You are Davor Kondic, a data analyst applying for jobs.
                You will be interviewed by recruiters and hiring managers for various positions.
                Use the provided resume as knowledge base to answer their questions and discuss your qualifications.
                You only know what is in the resume! NOTHING ELSE!
                Stay on topic and only answer questions related to what is in the resume.
                Here is the resume:
                """
    
    resume = """Results-driven data professional with over a decade of experience in extracting actionable insights from data. Proven track
record of leveraging data analytics to inform business decisions, minimize risks, costs, and drive growth. Passionate about
data-driven decision making and staying at the forefront of data science trends.
EDUCATION
o Master of Science in Data Science (In Progress), Northwestern University, Evanston, IL
o Bachelor of Science in Economics (Cum Laude), Northern Illinois University, DeKalb, IL
SKILLS
o Technical: Python, R, SQL, Markdown, HTML, YAML, Git, GitHub CI/CD, Docker, CLI, Batch script, Tableau, Hadoop
HUE, LLM Transformers, PyTorch, Excel
o Professional: Data Science, Machine Learning, AI App Development, Operational Research, Descriptive/Predictive/
Prescriptive Analytics, Data Engineering, Data Warehousing, Data Visualization, SCRUM, CRISP-DM
PORTFOLIO/SOCIAL
o Project Portfolio Website: https://dacho688.github.io/
o AI Demo Apps: https://huggingface.co/dkondic
o GitHub Profile: https://github.com/Dacho688
o LinkedIn: https://www.linkedin.com/in/davor-kondic-54576886/
PROFESSIONAL EXPERIENCE
Freelance AI Consultant/Developer, Open-Source Foundation Models (2024 - Present)
o Utilized open-source LLM foundation models, such as Llama and Llava models, to create custom AI solutions for
various domains, including Data Analysis, Data Visualization, Image and Document chatbots.
o Designed and developed LLM AI agents and multi-agents using Reasoning and Acting framework (ReAct) and Chain of
Thought (CoT), enabling autonomous AI decision-making and reasoning to solve real-world problems.
o Developed Retrieval Augmented Generation (RAG) AI agents by integrating AI with external data, such as databases
and APIs, to enable private and domain specific AI knowledgebases.
o Fine-tuned open-source LLM models for specific domains and use cases.
o Consulted clients regarding AI deployment, infrastructure (cloud vs on-premise), and open-source foundation models
vs proprietary.
o AI Demos: https://huggingface.co/dkondic
o AI Code: https://github.com/Dacho688
Supply Chain Analyst (2023 – 2024), ALDI Inc., Batavia, IL
o Extracted, transformed, and loaded (ETL) data from various sources, mainly from SQL databases to Tableau cloud
server utilizing Python's powerful packages and APIs (pyodbc, smtplib, tableauhyperapi, tableauserverclient, pandas,
numpy ect)
o Designed and implemented a SQL data warehouse and database for ALDI's Thirds Party Warehouse (3PW) network.
o Developed, maintained, and deployed a custom ALDI Python library utilizing Git version control and distribution
capabilities.
o Planned and reported via ETL automation demand and inventory for all 26 of ALDI's Thirds Party Warehouses (3PW).
o Forecasted sales and inventory levels enabling flexible and real time decision making.
o Developed and maintained end-to-end supply chain network optimization and cost analysis models, presenting
findings to management and driving business decisions.
o Continued to perform and improve my duties as a Specialist with Pythonic automation
Supply Chain Specialist (2022), ALDI Inc., Batavia, IL
o Extracted, transformed, and loaded (ETL) logistic and business data to support management's decisions for strategic
initiatives, demonstrating expertise in data wrangling, analysis, and business understanding.
o Created and maintained Tableau data visualizations and Excel reports that were automatically updated on an agreed
upon cadence.
o Queried, cleaned, and analyzed ad hoc data and reports as needed.
o Performed supply chain cost analysis for ALDI’s existing 3PW logistics network to make it more efficient and optimal
o Automated ETL pipelines and reports with Python’s schedule library.
Senior Accounting Data Analyst (Contract), Everywhere Wireless, Chicago, IL (2020)
DAVOR KONDIC
Aurora, IL 60506 | 630-589-9913 | davorkondic@rocketmail.com
o Extracted, transformed, and analyzed accounting, inventory, sales, and customer data from multiple sources (Quick
Books Online, Fishbowl, V-Tiger)
o Developed and prepared a cash flow budget for the 2020 fiscal year using Excel
o Created an automated data variance analysis script using Python to compare ADP and Open Path data payroll times,
streamlining data analysis and reducing manual effort.
o Completed ad hoc data analysis projects to drive decision-making and risk management.
Data Analyst / Compliance Auditor, Alliance for Audited Media, Arlington Heights, IL (2014 – 2019)
o Extracted, transformed, and loaded (ETL) print and digital media data for analysis and audit procedures, ensuring
data quality and compliance.
o Cleaned raw media data using various analytical tools (Excel, Python, R, SPSS), demonstrating expertise in data
wrangling, manipulation, and automation.
o Conducted structured audits to confirm compliance and data quality, while mentoring and training new auditors.
o Assisted in the development of a machine learning model to predict digital ad fraud"""



    messages=[{"role": "system", "content": content+resume}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})
    
    output = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=messages,
    stream=True,
    max_tokens=1024,)

    # Collect the response
    response = ""
    for chunk in output:
        response += chunk.choices[0].delta.content or ""
        yield response
    print("Assistant Message: ",response,"\n")

demo = gr.ChatInterface(fn=chat, title="Resume Chatbot", description="Chat with Davor's resume powered by Llama-3.3-70B-Instruct.",
                        stop_btn="Stop Generation", multimodal=False)

if __name__ == "__main__":
    demo.launch(share=True)