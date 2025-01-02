import streamlit as st
import os
import toml
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from phi.agent import Agent
from phi.model.groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Contact information constants
SUPPORT_EMAIL = "support@myayurhealth.com"
SUPPORT_PHONE = "+1 (555) 123-4567"

@dataclass
class DocumentResponse:
    content: str
    confidence: float
    metadata: Dict
    is_doctor_info: bool = False

class VectorDBService:
    def __init__(self, api_url: str = None, api_key: str = None):
        try:
            if api_url and api_key:
                self.client = QdrantClient(url=api_url, api_key=api_key)
            else:
                self.client = QdrantClient(":memory:")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.collection_name = "myayurhealth_docs"
        except Exception as e:
            error_msg = f"""Vector DB Initialization Error: {str(e)}
            Please contact our support team for assistance:
            Email: {SUPPORT_EMAIL}
            Phone: {SUPPORT_PHONE}"""
            st.error(error_msg)
            self.client = None
            self.model = None
    
    def search(self, query: str, limit: int = 5) -> List[DocumentResponse]:
        if not self.client or not self.model:
            return []
        
        try:
            query_vector = self.model.encode(query).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                DocumentResponse(
                    content=result.payload.get('text', ''),
                    confidence=float(result.score),
                    metadata=result.payload.get('metadata', {}),
                    is_doctor_info='doctor' in result.payload.get('metadata', {}).get('type', '').lower()
                )
                for result in results
            ]
        except Exception as e:
            error_msg = f"""Search Error: {str(e)}
            Please contact our support team for assistance:
            Email: {SUPPORT_EMAIL}
            Phone: {SUPPORT_PHONE}"""
            st.error(error_msg)
            return []

class AyurvedaExpertSystem:
    def __init__(self, config: Dict[str, str]):
        self.vector_db = VectorDBService(
            api_url=config.get("QDRANT_URL"),
            api_key=config.get("QDRANT_API_KEY")
        )
        self.model = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            stream=True,
            description="Expert Ayurvedic healthcare assistant",
            instructions=[
                "Provide accurate Ayurvedic information based on available documentation",
                "Only recommend doctors that are explicitly mentioned in the documentation",
                "For health issues, explain Ayurvedic treatment approaches and recommend relevant doctors",
                "Be clear when information comes from documentation versus general knowledge"
            ]
        )
    
    def process_doctor_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        docs = self.vector_db.search(query)
        doctor_docs = [doc for doc in docs if doc.is_doctor_info]
        
        if not doctor_docs:
            return (f"I apologize, but I couldn't find any doctors matching your query in our platform. "
                   f"Please try a different search or contact our support team for assistance:\n"
                   f"Email: {SUPPORT_EMAIL}\nPhone: {SUPPORT_PHONE}", [])
        
        context = "\n".join([doc.content for doc in doctor_docs])
        response = self.model.run(f"""
        Based on the following doctor information from our platform, provide a clear response:
        {context}
        
        Format the response to clearly list each available doctor with their specializations and qualifications.
        
        End the response with:
        For appointments and inquiries, please contact our support team:
        Email: {SUPPORT_EMAIL}
        Phone: {SUPPORT_PHONE}
        """).content
        
        return response, doctor_docs

    def process_health_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        # First search for condition-specific information
        condition_docs = self.vector_db.search(query)
        
        # Then search for relevant doctors
        doctor_docs = self.vector_db.search(f"doctor treating {query}")
        doctor_docs = [doc for doc in doctor_docs if doc.is_doctor_info]
        
        all_docs = condition_docs + doctor_docs
        
        if not all_docs:
            # Generate a general response if no specific documentation is found
            response = self.model.run(f"""
            Provide information about how Ayurveda approaches treating {query}. 
            Include:
            1. The Ayurvedic perspective on this condition
            2. General treatment principles
            3. Note that this is general information and specific treatment should be discussed with an Ayurvedic practitioner
            
            End with:
            For personalized consultation and treatment, please contact our support team:
            Email: {SUPPORT_EMAIL}
            Phone: {SUPPORT_PHONE}
            """).content
            return response, []
        
        # Combine documented information with doctor recommendations
        context = "\n".join([doc.content for doc in all_docs])
        response = self.model.run(f"""
        Based on the following information from our platform, provide a comprehensive response about {query}:
        {context}
        
        Include:
        1. The Ayurvedic approach to treating this condition
        2. Specific treatments or therapies mentioned in our documentation
        3. Available doctors who specialize in treating this condition
        
        Only mention doctors explicitly listed in the provided information.
        
        End with:
        For appointments and detailed treatment plans, please contact our support team:
        Email: {SUPPORT_EMAIL}
        Phone: {SUPPORT_PHONE}
        """).content
        
        return response, all_docs

    def process_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        # Check if query is about doctors
        if any(keyword in query.lower() for keyword in ['doctor', 'practitioner', 'physician', 'vaidya']):
            return self.process_doctor_query(query)
        
        # Check if query is about health conditions
        elif any(keyword in query.lower() for keyword in ['treat', 'cure', 'healing', 'medicine', 'therapy', 'disease', 'condition', 'problem', 'pain']):
            return self.process_health_query(query)
        
        # General query
        docs = self.vector_db.search(query)
        if not docs:
            response = self.model.run(f"""
            Provide accurate general information about {query} from an Ayurvedic perspective.
            Note that this is general knowledge and specific health advice should be sought from qualified practitioners.
            
            For personalized consultation and advice, please contact our support team:
            Email: {SUPPORT_EMAIL}
            Phone: {SUPPORT_PHONE}
            """).content
            return response, []
        
        context = "\n".join([doc.content for doc in docs])
        response = self.model.run(f"""
        Based on the following information from our documentation, provide a response about {query}:
        {context}
        
        Ensure the response is accurate and based only on the provided information.
        
        End with:
        For more information and personalized guidance, please contact our support team:
        Email: {SUPPORT_EMAIL}
        Phone: {SUPPORT_PHONE}
        """).content
        
        return response, docs

def load_config():
    config = {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": ""
    }
    
    try:
        with open("secrets.toml", "r") as f:
            toml_config = toml.load(f)
            config.update(toml_config)
    except FileNotFoundError:
        st.warning(f"""secrets.toml not found. Using default or environment variables.
        If you continue to experience issues, please contact our support team:
        Email: {SUPPORT_EMAIL}
        Phone: {SUPPORT_PHONE}""")
    
    for key in config:
        env_value = os.getenv(key)
        if env_value:
            config[key] = env_value
    
    return config

def main():
    st.set_page_config(page_title="Ayurveda Expert System", layout="wide")
    st.title("Ayurveda Expert System")
    
    config = load_config()
    expert_system = AyurvedaExpertSystem(config)
    
    query = st.text_input("What would you like to know about Ayurvedic healthcare?")
    
    if st.button("Submit"):
        if not query:
            st.warning("Please enter a query.")
            return
            
        with st.spinner("Processing your query..."):
            response, docs = expert_system.process_query(query)
            
            # Display main response
            st.markdown("### Response")
            st.write(response)
            
            # Display source documents if available
            if docs:
                st.markdown("### Supporting Information")
                for idx, doc in enumerate(docs, 1):
                    with st.expander(f"Source {idx} (Confidence: {doc.confidence:.2f})"):
                        st.write(doc.content)
                        if doc.metadata:
                            st.markdown("**Metadata:**")
                            st.json(doc.metadata)
    
    # Add contact information footer
    st.markdown("---")
    st.markdown(f"""
    ### Need Help?
    Contact our support team:
    - Email: {SUPPORT_EMAIL}
    - Phone: {SUPPORT_PHONE}
    """)

if __name__ == "__main__":
    main()
