from llama_index.llms.gemini import Gemini
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
#
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import re
import json
import nest_asyncio
import pymupdf4llm
from llama_index.core import SummaryIndex, VectorStoreIndex
from langchain_google_vertexai import VertexAIEmbeddings
PROJECT_ID = 'white-watch-436805-d1'
REGION = 'us-central1'
MODEL_ID = 'text-embedding-004'
import os
class DocumentProcessor:
    def __init__(self, pdf_url="2020-21CyberSecurity.pdf"):
        self.pdf_url = pdf_url
        self.document = ""
        self.nodes = None
        self.full_text=None
    def process_to_text(self):
        md_text = pymupdf4llm.to_markdown(self.pdf_url)
        self.full_text=md_text
        # self.document = Document(text=md_text)
        # splitter = SentenceSplitter(chunk_size=1024)
        # self.nodes = splitter.get_nodes_from_documents([self.document])
    def process_to_nodes(self):
        self.document = Document(text=self.full_text)
        splitter = SentenceSplitter(chunk_size=2000,chunk_overlap=200)
        self.nodes = splitter.get_nodes_from_documents([self.document])        

class EmbeddingModel:
    @staticmethod
    def get_embedding_model():
        # lc_embed_model = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-mpnet-base-v2"
        # )
        embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

        return embeddings#LangchainEmbedding(lc_embed_model)

class IndexBuilder:
    def __init__(self, nodes):
        self.nodes = nodes
        self.summary_index = None
        self.vector_index = None

    def build_indices(self):
        self.summary_index = SummaryIndex(self.nodes)
        self.vector_index = VectorStoreIndex(self.nodes)

class QueryEngineBuilder:
    def __init__(self, summary_index, vector_index):
        self.summary_index = summary_index
        self.vector_index = vector_index
        self.query_engine = None

    def build_query_engine(self):
        summary_query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = self.vector_index.as_query_engine(vector_store_query_mode="mmr",vector_store_kwargs={"mmr_threshold":0.2})

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarization questions related to the cybersecurity audit report"
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
             description="Useful for question answering from the cybersecurity audit report"
        )

        self.query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=True
        )

class CybersecurityFindingsExtractor:
    def __init__(self):
         self.llm = Gemini()
         Docobj=DocumentProcessor()
         Docobj.process_to_text()
         self.ref_text=Docobj.full_text
     
    def extract_findings(self):
        prompt = """
        Analyze a cybersecurity report and provide the key findings in a structured format. 
        The output should be in JSON format with a well-structured hierarchy. 
        Each main category should have subcategories and descriptions.

        Report content:\n
        """+self.ref_text+"""


        Analyze the following cybersecurity report and provide key findings in JSON format. The response must adhere to a clear hierarchical structure, focusing on the following categories:

        Threat Landscape: Overview of emerging threats and attack vectors, along with their impact.
        Vulnerabilities: List of top vulnerabilities identified (without severity ratings).
        Incident Response: Summary of recent incidents and evaluation of response strategies.
        Emerging Technologies: How new technologies are affecting cybersecurity and associated risks.
        Compliance and Regulatory Issues: Current compliance challenges and recent regulatory updates.
        Ensure that the JSON response is well-structured, and each category is described clearly and concisely. The findings should be actionable and relevant for cybersecurity professionals.

        
        Format the output as a JSON list of objects, each ALWAYS containing ONE EACH of:
        {
        "Threat Landscape": {
            "Emerging Threats": {
                "description": "Overview of new and evolving cyber threats",
                "examples": ["Threat 1", "Threat 2"],
                "impact": "Potential impact on systems and data"
            },
            "Attack Vectors": {
                "common_methods": ["Method 1", "Method 2"],
                "trends": "Recent trends in how attacks are executed"
            }
            },
        "Vulnerabilities": {
            "Critical Issues": {
                "top_vulnerabilities": ["Vulnerability 1", "Vulnerability 2"]
                }
            },
        "Incident Response": {
            "Recent Incidents": {
                "description": "Analysis of recent incidents and attack patterns",
                "response_effectiveness": "Evaluation of response strategies"
                }
            },
        "Emerging Technologies": {
            "Impact": {
                "description": "How new technologies are affecting cybersecurity",
                "associated_risks": ["Risk 1", "Risk 2"]
                }
            },
        "Compliance and Regulatory Issues": {
            "Challenges": {
                "description": "Current compliance challenges faced",
                "regulatory_updates": "Recent regulatory updates"
                }
            }
        }
        Provide a comprehensive analysis that a cybersecurity professional would find informative and actionable.
        """

        response = self.process_prompt(prompt)
        json_match = re.search(r"\[([\s\S]*?)\]$", response)
        if json_match:
            json_str = json_match.group(1)
            #print(json_str)       
        else:
            json_str = response
        return self.extract_and_validate_json(json_str)

    def process_prompt(self, prompt):
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while processing the prompt: {e}"

    def extract_and_validate_json(self, text):
        pattern = r"\[.*\]"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            json_text = matches[-1]  
            try:
                json_data = json.loads(json_text)
                # if self.validate_json_structure(json_data):
                return json_data
                # else:
                #     return "JSON structure is not sufficiently detailed or well-structured."
            except json.JSONDecodeError:
                return f" JSON not proper in the output.{text}"
        else:
            return f"No JSON found in the output.{text}"

    # def validate_json_structure(self, json_data):
    #     if not isinstance(json_data, dict) or len(json_data) < 3:
    #         return False
    #     for value in json_data.values():
    #         if not isinstance(value, dict) or len(value) < 1:
    #             return False
    #         for subvalue in value.values():
    #             if not isinstance(subvalue, (dict, list, str)):
    #                 return False
    #     return True

##################################
##################################

class VulnerabilityExtractorAndRanker:
    def __init__(self, cybersecurity_findings):
        self.llm = Gemini()
        self.cybersecurity_findings = cybersecurity_findings

    def extract_and_rank_vulnerabilities(self):
        prompt = f"""
        Analyze the following cybersecurity findings and extract a list of vulnerabilities. 
        Then, rank these vulnerabilities from most critical to least critical.

        Cybersecurity Findings:
        {json.dumps(self.cybersecurity_findings, indent=2)}

        For each vulnerability:
        1. Provide a brief description
        2. Assign a criticality score from 1-10 (10 being most critical)
        3. Explain the reasoning behind the criticality score

        Format the output as a JSON list of objects, each ALWAYS containing ONE EACH of:
        - "vulnerability": (string) Name or brief description of the vulnerability
        - "criticality_score": (number) Score from 1-10
        - "reasoning": (string) Explanation for the score
        - "mitigation": (string) Brief suggestion for mitigation

        Sort the list by criticality_score in descending order.

        Example:
        [
          {{
            "vulnerability": "Unpatched software with known exploits",
            "criticality_score": 9,
            "reasoning": "Easily exploitable and can lead to full system compromise",
            "mitigation": "Implement regular patching schedule and vulnerability scanning"
          }},
          ...
        ]
        """

        response = self.llm.complete(prompt)
        return self.parse_vulnerabilities(response.text)

    def parse_vulnerabilities(self, response):
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                vulnerabilities = json.loads(json_match.group())
                return sorted(vulnerabilities, key=lambda x: x['criticality_score'], reverse=True)
            except json.JSONDecodeError:
                return "Error: Unable to parse the vulnerability list as JSON."
        else:
            return "Error: No valid JSON found in the response."

#############################
#############################
class DocumentQuerySystem:
    def __init__(self, pdf_url):
        nest_asyncio.apply()
        Settings.llm = Gemini()
        Settings.embed_model = EmbeddingModel.get_embedding_model()

        self.document_processor = DocumentProcessor(pdf_url)
        self.document_processor.process_to_text()
        self.document_processor.process_to_nodes()

        self.index_builder = IndexBuilder(self.document_processor.nodes)
        self.index_builder.build_indices()

        self.query_engine_builder = QueryEngineBuilder(
            self.index_builder.summary_index,
            self.index_builder.vector_index
        )
        self.query_engine_builder.build_query_engine()
    def summarize(self):
        llm = Gemini()
        
        summarization_prompt = f"""
        Please provide a comprehensive summary of the following document. 
        The summary should:
        1. Capture the main topics and key points discussed in the document
        2. Highlight any significant findings or conclusions
        3. Mention any important recommendations or action items
        4. Be concise yet informative, aiming for about 500 words

        Document to summarize:
        {self.document_processor.full_text}

        Summary:
        """

        response = llm.complete(summarization_prompt)
        return response.text
     
    def query(self, question):
        return self.query_engine_builder.query_engine.query(question)
if __name__ == "__main__":
    pdf_url = "2020-21CyberSecurity.pdf"
    query_system = DocumentQuerySystem(pdf_url)
    #response = query_system.summarize()
    #print(str(response))
    # Cyber_obj=CybersecurityFindingsExtractor()
    # findings=Cyber_obj.extract_findings()
    # print(json.dumps(findings,indent=2))
    # extractor = VulnerabilityExtractorAndRanker(findings)
    # ranked_vulnerabilities = extractor.extract_and_rank_vulnerabilities()
    
    # print(json.dumps(ranked_vulnerabilities, indent=2))
    # llm=Gemini()
    # Doc=Cyber_obj.ref_text
    # response=llm.complete()
    # print(response)
    hello=input()
    print(query_system.query(f"""[SYSTEM MESSAGE]
                                         You are now adopting the persona of Fischer, a knowledgeable and friendly AI assistant
                                         from the CyberStrike AI Audit Management Suite. Your primary role is to assist users in
                                         navigating cybersecurity audit processes, providing insights, and enhancing the overall 
                                         quality of audit reports. Emphasize your expertise in risk assessments, compliance, 
                                         vulnerability analysis, and remediation recommendations. Be personable, approachable, 
                                         and solution-oriented.
                                         ONLY SAY "Hi there! I'm Fischer, your friendly AI assistant from CyberStrike." ONCE. IF THE USER QUERY CONTAINS YOUR GREETING DON'T GREET THE USER AGAIN
                             
                                    #     [USER INPUT]:{hello}
                                    # """))
