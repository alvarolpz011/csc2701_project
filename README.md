# csc2701_project

## Problem Statement
RAG (Retrieval Augmented Generation) is one of the many techniques that the recent developments in the area of Natural Language Processing, specifically Larga Language Models, have made possible. Unlike traditionally depending on language models that rely solely on their internal knowledge with which they were trained on, RAG combines the generative properties of LLMs with information retrieval systems to generate more accurate and contextually relevant responses based on real information. This approach addresses the limitations of  knowledge in LLMs by dynamically extracting and incorporating up-to-date or domain-specific information to their context. As a result, RAG has gained attention for applications in open-domain question answering, enterprise information search, healthcare, and more. However, developing and deploying an effective RAG system presents challenges that need to be overcome with effective ML Ops practices.

(missing mention of why we're doing it with MSCAC information for students)


## Data Sources & Extraction
Given the nature of RAG systems, data sources can vary depending on the applications. Commonly, data sources can include:
* Text documents: Articles, reports, books, etc. Common formats can include PDF, DOCX, TXT, HTML, etc.
* Databases: Structured data from SQL or NoSQL databases.
* APIs: Data from web services or other online sources.
* Multimedia, such as images, audio, and video.
* Other knowledge bases like webpages, wikis, etc.

For this specific project, we will be using a collection of informational PDF documents aimed at MSCAC students. These documents include:
* MSCAC 2025-2026 Student Handbook
* The MScAC [Program Webpage](https://mscac.utoronto.ca/)
* The CS 2025-2026 Fall/Winter [Graduate Course Timetables Webpage ](https://web.cs.toronto.edu/graduate/timetable)

In all of these sources, the data must be cleaned and preprocessed to ensure that it is in a suitable format for the RAG system to produce accurate and relevant responses. In the case of the PDF documents, text extraction techniques will be needed to convert its content into structured text data easily understandable by the RAG system generativve model. As for the webpages, web scraping techniques will be needed to extract the desired information from the HTML pages, which could be found as plain text, tables, lists, images, etc.

The data extraction process will need a defined pipeline to deal with these data formats. The following steps will be taken:
1. **Data Collection**: Gather all relevant documents and webpages via an automated web scraper or manual download.
2. **Text Extraction**: Use specialized libraries (PyMuPDF, PyPDF2, pdfplumber, etc.) to extract text from PDF documents. For webpages, options include BeautifulSoup, Scrapy, Selenium, or even AI-based web scrappers.
3. **Data Cleaning**: Remove any unnecessary elements, normalize the text by converting it to lowercase and removing extra spaces.
4. **Text Chunking**: Split the text into smaller, manageable chunks for processing before being saved into a database or vector store.
5. **Data Storage**: Store the cleaned and chunked text data in a suitable format, such as a database or a vector store, for easy retrieval during the RAG process.

## Plan for EDA and modeling

[RAG Systems Techniques List](https://github.com/NirDiamant/RAG_Techniques)


## Initial plan for data exploration and modeling

### Data exploration


<img width="1728" height="1452" alt="image" src="https://github.com/user-attachments/assets/8d0c748f-0800-4632-af58-6cb28d27030a" />


### RAG pipeline 
1. Parsing text data from the data source.
2. Data preprocessing: removing redundant structural emlements like headers/footers, normalizing text 
2. Chunking. Define the chunking strategy - it can be  a fixed-size, a recursive or a context-aware chunking.
3. Building embeddings. Decide between using sparse embeddings for keyword retrieval  (like BM25 or TF-IDF) or dense embeddings (like SBERT) to capture semantic meaning. Or use hybrid search and combine these approaches.
4. Retriever algorithm
5. Reranker algorithm
6. Response generation using top-k chunks and LLM
   
