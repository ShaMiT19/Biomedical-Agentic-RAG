from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Optional, Dict, Any, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.cache import InMemoryCache
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os, re, requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from xml.etree import ElementTree as ET
import streamlit as st

load_dotenv()

def get_secret(key):
    """Robustly retrieve secrets from Streamlit secrets (Cloud/Local) or Environment."""
    try:
        # Try direct access (Streamlit Cloud Dashboard)
        if key in st.secrets:
            return st.secrets[key]
        # Try nested access (local secrets.toml with [secrets] section)
        if "secrets" in st.secrets and key in st.secrets["secrets"]:
            return st.secrets["secrets"][key]
    except Exception:
        pass
    # Fallback to env var
    return os.getenv(key)

# Set into environment for LangChain tools and clients to pick up automatically
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
OPENFDA_API_KEY = get_secret("OPENFDA_API_KEY")

if OPENAI_API_KEY: os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TAVILY_API_KEY: os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if OPENFDA_API_KEY: os.environ["OPENFDA_API_KEY"] = OPENFDA_API_KEY

# Initialize LLMs
cache = InMemoryCache()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=OPENAI_API_KEY, cache=cache)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

def get_tavily_tool():
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        # Attempt initialization. LangChain tools ideally pick up the ENV variable.
        return TavilySearchResults(max_results=2)
    except Exception:
        try:
            # Fallback for older package versions or specialized strict Pydantic models
            return TavilySearchResults(tavily_api_key=os.environ["TAVILY_API_KEY"], max_results=2)
        except Exception:
            return None

tavily_search_tool = get_tavily_tool()

# Define Pydantic models
class PatientProfile(BaseModel):
    age: int = Field(..., gt=0, le=120, description="Age of the patient")
    gender: str = Field(..., description="Gender of the patient")
    height: float = Field(..., gt=0, description="Height of the patient")
    weight: float = Field(..., gt=0, description="Weight of the patient")
    systolic_bp: float = Field(..., gt=0, le=200, description="Systolic blood pressure of the patient")
    diastolic_bp: float = Field(..., gt=0, le=100, description="Diastolic blood pressure of the patient")
    cholesterol: float = Field(..., gt=0, le=500, description="Cholesterol level of the patient")
    ldl_cholesterol: float = Field(..., gt=0, le=500, description="LDL cholesterol level of the patient")
    hdl_cholesterol: float = Field(..., gt=0, le=500, description="HDL cholesterol level of the patient")
    triglycerides: float = Field(..., gt=0, le=500, description="Triglycerides level of the patient")
    heartbeat_rate: float = Field(..., gt=0, le=200, description="Heartbeat rate of the patient")
    temperature: float = Field(37.0, gt=30, le=45, description="Body temperature in Celsius")
    respiratory_rate: int = Field(16, gt=0, le=60, description="Respiratory rate in breaths per minute")
    wbc_count: float = Field(7.0, gt=0, le=100.0, description="White Blood Cell count (x10^9/L)")
    platelets: float = Field(250.0, gt=0, le=1000.0, description="Platelet count (x10^9/L)")
    oxygen_saturation: float = Field(98.0, gt=0, le=100.0, description="Oxygen saturation SpO2 (%)")
    sugar_level: float = Field(..., gt=0, le=20, description="Sugar level of the patient")
    avg_sleep_hours: float = Field(..., gt=0, le=24, description="Average sleep hours of the patient")
    avg_daily_steps: float = Field(..., gt=0, le=20000, description="Average daily steps of the patient")
    symptoms: List[str] = Field(default_factory=list, description="Symptoms of the patient")
    medical_history: List[str] = Field(default_factory=list, description="Medical history of the patient")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Allergies of the patient")
    immunizations: Optional[List[str]] = Field(default_factory=list, description="Immunizations of the patient")
    
    @computed_field
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)
    
    @computed_field
    def pulse_pressure(self) -> float:
        return self.systolic_bp - self.diastolic_bp
    
    @model_validator(mode="after")
    def validate_computed_metrics(self):
        if self.bmi <= 0:
            raise ValueError("BMI must be greater than 0")
        if self.pulse_pressure <= 0:
            raise ValueError("Pulse pressure cannot be negative")
        return self
    
class DiseaseRisk(BaseModel):
    disease_name: str = Field(..., description="Name of the disease")
    risk_score: float = Field(..., description="Risk score from 0.0 to 1.0")

class RiskAssessment(BaseModel):
    disease_risks: List[DiseaseRisk] = Field(default_factory=list, description="Disease risks associated with the patient")
    risk_flags: List[str] = Field(default_factory=list, description="Risk flags associated with the patient")
    risk_summary: str = Field(..., description="Summary of the risk assessment")

class RiskResponse(BaseModel):
    identified_risks: List[DiseaseRisk] = Field(..., description="List of identified disease risks and their scores")
    risk_flags: List[str] = Field(..., description="High-priority clinical risk flags")
    risk_summary: str = Field(..., description="Professional risk summary")

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dose: str = Field(..., description="Dosage and frequency")
    mechanism: str = Field(..., description="Short mechanism of action")
    notes: str = Field(..., description="Monitoring and safety notes")

class PrescriptionPlan(BaseModel):
    medications: List[Medication] = Field(default_factory=list, description="Medications to be prescribed")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for the patient")
    instructions: List[str] = Field(default_factory=list, description="Instructions for the patient")

class LifestylePlan(BaseModel):
    exercises: List[str] = Field(default_factory=list, description="Exercise recommendations for the patient")
    diet: List[str] = Field(default_factory=list, description="Diet recommendations for the patient")
    sleep: List[str] = Field(default_factory=list, description="Sleep recommendations for the patient")
    metabolic_advice: List[str] = Field(default_factory=list, description="Metabolic advice for the patient")
    
class MedicalSearchQuery(BaseModel):
    query: str = Field(..., description="Generated medical search query")
    
class DocEvalScore(BaseModel):
    score: float = Field(..., description="Relevancy score for the document")
    reason: str = Field(..., description="Reason for the score")
    
class MedicalEvidence(BaseModel):
    query: str = Field(..., description="Generated medical search query")
    retrieved_chunks_count: int = Field(..., description="Number of chunks retrieved")
    refined_context: str = Field(..., description="Filtered evidence-based sentences")
    clinical_summary: str = Field(..., description="LLM-synthesized clinical summary")
    sources_used: Optional[List[str]] = Field(default_factory=list, description="Sources used (PubMed, Tavily, etc.)")
    
class ClinicalAlert(BaseModel):
    urgency: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] = Field(..., description="Urgency level of the alert")
    message: str = Field(..., description="Clinician-facing alert message")
    conditions_flagged: Optional[List[str]] = Field(default_factory=list, description="List of diseases that triggered escalation")
    interaction_flags: Optional[List[str]] = Field(default_factory=list, description="Detected multi-modibidity conditions and risk interaction patterns")
    recommended_action: str = Field(..., description="Recommend next clinical action based on urgency level.")
    
class AgentState(BaseModel):
    patient_profile: PatientProfile = Field(..., description="Patient profile")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    prescription_plan: PrescriptionPlan = Field(..., description="Prescription plan")
    lifestyle_plan: LifestylePlan = Field(..., description="Lifestyle plan")
    raw_patient_data: Dict[str, Any] = Field(default_factory=dict, description="Raw patient data")
    medical_search_query: MedicalSearchQuery = Field(..., description="Medical search query based on patient profile")
    medical_evidence: MedicalEvidence = Field(..., description="Medical evidence retrieved from external knowledge sources")
    clinical_alert: ClinicalAlert = Field(default=None, description="Clinical escalation alerts triggered by high severity risk conditions")
    treatment_road_map: str = Field(default="", description="Consolidated clinical treatment strategy and road map")
    
# Define the node functions
def collect_patient_data(state: AgentState) -> dict:
    """   
        Collects patient data from raw structured inputs and converts it into a PatientProfile object.
    """
    raw_patient_data = state.raw_patient_data
    patient = state.patient_profile.model_copy(deep=True)
    
    # Symptoms & History (EHR)
    patient.age = raw_patient_data.get("age", None)
    patient.gender = raw_patient_data.get("gender", None)
    patient.height = raw_patient_data.get("height", None)
    patient.weight = raw_patient_data.get("weight", None)
    patient.systolic_bp = raw_patient_data.get("systolic_bp", None)
    patient.diastolic_bp = raw_patient_data.get("diastolic_bp", None)
    patient.cholesterol = raw_patient_data.get("cholesterol", None)
    patient.triglycerides = raw_patient_data.get("triglycerides", None)
    patient.heartbeat_rate = raw_patient_data.get("heartbeat_rate", None)
    patient.temperature = raw_patient_data.get("temperature", 37.0)
    patient.respiratory_rate = raw_patient_data.get("respiratory_rate", 16)
    patient.wbc_count = raw_patient_data.get("wbc_count", 7.0)
    patient.platelets = raw_patient_data.get("platelets", 250.0)
    patient.oxygen_saturation = raw_patient_data.get("oxygen_saturation", 98.0)
    patient.sugar_level = raw_patient_data.get("sugar_level", None)
    patient.symptoms = raw_patient_data.get("symptoms", [])
    patient.medical_history = raw_patient_data.get("medical_history", [])
    patient.immunizations = raw_patient_data.get("immunizations", [])
    patient.allergies = raw_patient_data.get("allergies", [])
    patient.avg_sleep_hours = raw_patient_data.get("avg_sleep_hours", None)
    patient.avg_daily_steps = raw_patient_data.get("avg_daily_steps", None)
    return {"patient_profile": patient}

def early_disease_detection(state: AgentState) -> dict:
    """   
    Detects early signs of ANY disease (Oncology, Infectious, Chronic, etc.).
    """
    patient = state.patient_profile
    
    detection_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are an advanced clinical diagnostic support system.
            Your task is to analyze patient biometrics, symptoms, and medical history to identify ALL potential disease risks.
            
            Coverage includes but is not limited to:
            - Chronic (Diabetes, Cardiovascular, Hypertension, Metabolic)
            - Oncology (Tumors, Cancers - based on symptoms and markers like platelets/WBC)
            - Infectious (COVID-19, HIV/AIDS, Viral, Bacterial - based on temperature, respiratory rate, oxygen, WBC)
            - Autoimmune, Endocrine, and Neurological conditions.
            
            Rules:
            - Return a dictionary of disease names and their estimated risk scores (0.0 to 1.0).
            - Identify high-priority "risk_flags" for rapid clinical alerting.
            - Provide a concise, professional risk summary.
            - Do NOT diagnose.
            - Output strictly in JSON matching the provided structure.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Profile:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            BP: {sbp}/{dbp}, Pulse: {hr}, Temp: {temp}°C, Resp: {resp}, SpO2: {spo2}%
            WBC: {wbc}, Platelets: {plt}, Sugar: {sugar}, Lipids: LDL {ldl}/HDL {hdl}
            Symptoms: {symptoms}
            History: {history}
            
            Identify all relevant disease risks and provide a structured assessment.
            """
        )
    ])
    
    detection_chain = detection_prompt | llm.with_structured_output(RiskResponse)
    
    results = detection_chain.invoke({
        "age": patient.age,
        "gender": patient.gender,
        "bmi": patient.bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "hr": patient.heartbeat_rate,
        "temp": patient.temperature,
        "resp": patient.respiratory_rate,
        "spo2": patient.oxygen_saturation,
        "wbc": patient.wbc_count,
        "plt": patient.platelets,
        "sugar": patient.sugar_level,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "symptoms": ", ".join(patient.symptoms),
        "history": ", ".join(patient.medical_history)
    })
    
    risk_assessment = RiskAssessment(
        disease_risks=results.identified_risks,
        risk_flags=results.risk_flags,
        risk_summary=results.risk_summary
    )
    
    return {"risk_assessment": risk_assessment}

def clinical_triage_router(state: AgentState) -> Literal["high_risk", "low_risk"]:
    """
        Routes the workflow based on detected risk levels.
    """
    risks = state.risk_assessment.disease_risks
    if not risks:
        return "low_risk"
    
    # If any risk is >= 25% (0.25), consider it high/clinical risk
    max_risk = max([r.risk_score for r in risks])
    if max_risk >= 0.60:
        return "high_risk"
    
    return "low_risk"

def generate_medical_search_query(state: AgentState) -> dict:
    """
        Generates a PubMed search query based on the patient's risk assessment."""
    # Only include diseases with significant risk
    high_risk_diseases = [risk.disease_name for risk in state.risk_assessment.disease_risks if risk.risk_score >= 0.25]

    medical_query_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a clinical research assistant. "
            "Given a patient's clinical risk assessment, generate a **short list of disease/condition keywords** "
            "suitable for PubMed and clinical literature searches.\n"
            "Rules:\n"
            "- Only include the most relevant identified conditions (Chronic, Infectious, Oncology, etc.)\n"
            "- Return keywords separated by commas\n"
            "- Do NOT write full sentences\n"
            "- Output JSON matching: {{'query': 'condition1, condition2'}}"
        ),
        HumanMessagePromptTemplate.from_template(
            "High-risk diseases:\n{risks}\nGenerate PubMed search keywords."
        )
    ])

    medical_query_chain = medical_query_prompt | llm.with_structured_output(MedicalSearchQuery)

    output = medical_query_chain.invoke({
        "risks": ", ".join(high_risk_diseases) if high_risk_diseases else "general health"
    })

    return {"medical_search_query": output.query}

def pubmed_search(query: str, retmax: int = 5):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "email": "your_email@example.com"  # recommended
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.json()["esearchresult"]["idlist"]

def pubmed_fetch(pmids: list[str]):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.text

def parse_pubmed_xml(xml_text: str) -> list[Document]:
    root = ET.fromstring(xml_text)
    docs = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="")
        abstract = " ".join(
            [t.text or "" for t in article.findall(".//AbstractText")]
        )
        mesh_terms = [
            m.text for m in article.findall(".//MeshHeading/DescriptorName")
        ]

        content = f"{title}\n\n{abstract}"

        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "mesh_terms": mesh_terms,
                    "source": "PubMed"
                }
            )
        )

    return docs

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 25
    ]
    
def fetch_medical_literature(state: AgentState) -> dict:
    """ 
        Fetches relevant medical literature based on the generated search query.
        Combines internal medical knowledge + web-based guideline retrieval
        with strict relevance filtering.
    """
    query = generate_medical_search_query(state)
    search_query = query["medical_search_query"] if isinstance(query, dict) else query
    # Primary medical literature source
    try:
        pmids = pubmed_search(search_query)
        
        if pmids:
            pubmed_xml = pubmed_fetch(pmids)
            docs = parse_pubmed_xml(pubmed_xml)
        else:
            docs = []
            
        print(f"PubMed search returned {len(pmids)} results")
    except Exception as e:
        print(f"PubMed search/fetch failed: {e}")
        docs = []
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        return {
            "medical_evidence": MedicalEvidence(
                query=search_query,
                retrieved_chunks_count=0,
                refined_context="No specific literature findings for this query.",
                clinical_summary="No direct medical evidence was found in the queried sources for these combinations of risks.",
                sources_used=["PubMed"]
            )
        }
    
    # Clear irrelevant encoding characters like emojis and special characters
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        
    # Create embeddings for each chunk
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(search_query)
    
    # Evaluate relevance of retrieved chunks
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a strict retrieval evaluator for RAG based on medical knowledge."
            "Score each retrieved document in [0.0, 1.0] for relevance to the query.\n"
            "1.0 = most relevant, 0.0 = least relevant.\n"
            "Strictly return JSON: {'score': float, 'reason': str}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {query}\nChunk: {chunk}"
        )
    ])
    
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
    
    # Corrective RAG thresholds
    LOWER_THRESHOLD = 0.3
    UPPER_THRESHOLD = 0.7

    evaluated_docs = []
    scores = []
    
    for doc in retrieved_docs:
        try:
            result = doc_eval_chain.invoke({"query": search_query, "chunk": doc.page_content})
            scores.append(result.score)
            
            if result.score > LOWER_THRESHOLD:
                evaluated_docs.append(doc)
        except:
            continue
    
    good_docs = evaluated_docs.copy()
    
    # Perform Corrective-RAG routing
    verdict = "AMBIGUOUS"
    
    if any(score > UPPER_THRESHOLD for score in scores):
        verdict = "CORRECT"
        
    if all(score < LOWER_THRESHOLD for score in scores):
        verdict = "INCORRECT"
    
    # Fallback: Web-based retrieval using TavilySearch if verdict is either INCORRECT or AMBIGUOUS
    if verdict in ["INCORRECT", "AMBIGUOUS"] and tavily_search_tool:
        try:
            web_results = tavily_search_tool.run(search_query)
            
            if isinstance(web_results, dict) and "results" in web_results:
                web_results = web_results["results"]
            elif isinstance(web_results, list):
                web_results = web_results
            else:
                web_results = []
        except Exception as e:
            print(f"Tavily search failed: {e}")
            web_results = []
    else:
        web_results = []

    retrieved_docs = []
    
    for res in web_results:
        title = res.get("title", "")
        url = res.get("url", "")
        content = res.get("content", "") or res.get("snippet", "")
        full_text = f"{title}\nURL: {url}\n\n{content}"
        
        retrieved_docs.append(
            Document(
                page_content=full_text.strip(),
                metadata={
                    "source": "Web/TavilySearch",
                    "url": url,
                    "title": title
                }
            )
        )
            
    if verdict == "CORRECT":
        final_relevant_docs = good_docs
    elif verdict == "INCORRECT":
        final_relevant_docs = retrieved_docs
    else: # AMBIGUOUS
        final_relevant_docs = good_docs + retrieved_docs
            
    # Sentence-level decomposition
    all_sentences = []
    
    for doc in final_relevant_docs:
        sentences = split_sentences(doc.page_content)
        all_sentences.extend(sentences)
        
    filtered_context = "\n".join(all_sentences)
    
    # Clinical synthesis
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a medical evidence synthesizer.\n"
            "Summarize the evidence in a clinical context based on the patient profile and medical literature.\n"
            "Do NOT hallucinate. Do NOT diagnose or prescribe.\n"
        ),
        HumanMessagePromptTemplate.from_template(
            "Context: {context}"
        )
    ])
    
    clinical_summary = llm.invoke(synthesis_prompt.invoke({"context": filtered_context})).content
    
    return {
        "medical_evidence": MedicalEvidence(
            query=search_query,
            retrieved_chunks_count=len(retrieved_docs),
            refined_context=filtered_context,
            clinical_summary=clinical_summary,
            sources_used=list({doc.metadata.get("source", "Unknown") for doc in retrieved_docs})
        )
    }
    
def check_openfda_warnings(drug_name: str) -> List[str]:
    """Query openFDA Label API for boxed warnings and warnings."""
    try:
        url = (
            f"https://api.fda.gov/drug/label.json"
            f"?api_key={OPENFDA_API_KEY}"
            f"&search=openfda.generic_name:{drug_name}"
            f"&limit=1"
        )
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        warnings = []
        if "results" in data:
            result = data["results"][0]
            if "boxed_warning" in result:
                warnings.extend(result["boxed_warning"])
            if "warnings" in result:
                warnings.extend(result["warnings"])
        return warnings
    except requests.exceptions.HTTPError as e:
        return [f"openFDA HTTP error: {str(e)}"]
    except Exception as e:
        return [f"openFDA API failure: {str(e)}"]

def get_rxcui(drug_name: str):
    """Get RxCUI identifier for a drug name."""
    try:
        url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        resp = requests.get(url, params={"name": drug_name}, timeout=5)
        resp.raise_for_status()
        rxcuis = resp.json().get("idGroup", {}).get("rxnormId", [])
        return rxcuis[0] if rxcuis else None
    except Exception:
        return None

def get_drug_classes_by_rxcui(rxcui: str):
    """Get RxClass drug classes for a given RxCUI."""
    try:
        url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
        resp = requests.get(url, params={"rxcui": rxcui}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        associations = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        classes = []
        for assoc in associations:
            class_info = assoc.get("rxclassMinConceptItem", {})
            class_name = class_info.get("className")
            class_type = class_info.get("classType")
            if class_name:
                classes.append(class_name.lower())
        return classes
    except Exception:
        return []

def prescribe_medications(state: AgentState) -> dict:
    """   
        Generates evidence-based prescriptions, instructions, and recommendations
        based on the medical evidence retrieved from external knowledge sources, 
        patient profile, and risk assessment.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    
    formatted_disease_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score*100,2)}%"
        for risk in risk_assessment.disease_risks
    )
    
    # Build structured clinical prescription prompt
    prescription_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
                You are an evidence-based clinical decision support system.

                Your task:
                Generate a SAFE and medically appropriate prescription plan.

                Rules:
                - Base decisions strictly on patient data and retrieved medical evidence.
                - Consider contraindications, age, allergies, BMI, comorbidities.
                - Include first-line guideline therapies only.
                - Include dosage, frequency, mechanism (short), and monitoring notes.
                - Include safety warnings and drug interaction considerations.
                - If lifestyle therapy is preferred over medication, state so clearly.
                - Do NOT hallucinate unsupported treatments.
                - Do NOT diagnose.
                - Output strictly in structured JSON matching PrescriptionPlan schema.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
                Patient Profile:
                Age: {age}
                Gender: {gender}
                BMI: {bmi:.2f}
                Blood Pressure: {sbp}/{dbp}
                LDL: {ldl}
                HDL: {hdl}
                Triglycerides: {tg}
                Sugar Level: {sugar}
                Sleep: {sleep}
                Steps: {steps}
                Symptoms: {symptoms}
                Medical History: {history}
                Allergies: {allergies}

                Disease Risks:
                {risks}

                Retrieved Clinical Evidence Summary:
                {evidence}

                Generate a world-class, medically rigorous prescription plan based on the retrieved evidence.
                - Medications: Provide specific drug names, precise dosages, frequencies, AND clinical rationales (mechanism of action, specific evidence support).
                - Instructions: Provide detailed administration guidelines, potential side effects to monitor, and contraindications.
                - Recommendations: Provide advanced adjunctive therapies, specific monitoring intervals for metabolic markers, and evidence-based lifestyle synergies.
                
                Recommendations must be state-of-the-art, medically rigorous, and highly personalized. Never use generic or redundant dummy values.
                Never return NULL or empty lists for medications, instructions, or recommendations. 
                Output strictly in JSON matching PrescriptionPlan schema.
            """
        )
    ])
    
    prescription_chain = prescription_prompt | llm.with_structured_output(PrescriptionPlan)
    
    prescriptions = prescription_chain.invoke({
        "age": patient.age,
        "gender": patient.gender,
        "bmi": patient.bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "tg": patient.triglycerides,
        "sugar": patient.sugar_level,
        "sleep": patient.avg_sleep_hours,
        "steps": patient.avg_daily_steps,
        "symptoms": ", ".join(patient.symptoms),
        "history": ", ".join(patient.medical_history),
        "allergies": ", ".join(patient.allergies),
        "risks": formatted_disease_risks,
        "evidence": medical_evidence.clinical_summary
    })
    
    # Remove medications conflicting with allergies
    if patient.allergies:
        filtered_medications = []
        
        for medication in prescriptions.medications:
            if not any(allergy.lower() in medication.name.lower() for allergy in patient.allergies):
                filtered_medications.append(medication)
                
        prescriptions.medications = filtered_medications
    
    if not prescriptions.medications:
        prescriptions.medications = [Medication(name="No immediate medication recommended", dose="N/A", mechanism="N/A", notes="Follow lifestyle and monitor parameters closely")]
    if not prescriptions.recommendations:
        prescriptions.recommendations = ["Follow lifestyle and monitor parameters closely"]
    if not prescriptions.instructions:
        prescriptions.instructions = ["Reassess after follow-up"]
    
    return {
        "prescription_plan": prescriptions
    }


def drug_safety_guardrail(state: AgentState) -> dict:
    """
        Rule-based medication safety validation integrating OpenFDA + RxClass checks.
    """
    prescriptions = state.prescription_plan
    patient = state.patient_profile

    unsafe_flags = []
    safe_medications = []

    # Check for patient-specific allergy conditions
    for medication in prescriptions.medications:
        if patient.allergies:
            if any(allergy.lower() in medication.name.lower() for allergy in patient.allergies):
                unsafe_flags.append(f"{medication.name} contraindicated due to allergy")
                continue
        safe_medications.append(medication)

    # Check for drug interactions via RxClass
    med_classes = {}
    for med in safe_medications:
        rxcui = get_rxcui(med.name)
        if rxcui:
            classes = get_drug_classes_by_rxcui(rxcui)
            med_classes[med.name] = classes
        else:
            med_classes[med.name] = []

    anticoagulants = [m for m, cls in med_classes.items() if any("anticoagulant" in c for c in cls)]
    nsaids = [m for m, cls in med_classes.items() if any("nonsteroidal" in c or "nsaid" in c for c in cls)]

    for a in anticoagulants:
        for n in nsaids:
            unsafe_flags.append(f"{a} + {n}: potential bleeding risk (class interaction)")

    # Existing patient-specific checks
    medication_names = [med.name.lower() for med in safe_medications]

    # Blood sugar
    if "metformin" in medication_names and patient.sugar_level < 4.0:
        unsafe_flags.append("Metformin unsafe in hypoglycemia")

    # Blood pressure
    if patient.systolic_bp < 100:
        for med in safe_medications:
            if "lisinopril" in med.name.lower():
                unsafe_flags.append("Lisinopril unsafe during hypotension")

    # Update prescription plan
    state.prescription_plan.medications = safe_medications

    # Update clinical alert with all interaction flags
    if state.clinical_alert:
        state.clinical_alert.interaction_flags.extend(unsafe_flags)
    else:
        state.clinical_alert = ClinicalAlert(
            urgency="LOW",
            message="Drug safety check completed",
            conditions_flagged=[],
            interaction_flags=unsafe_flags,
            recommended_action="Review unsafe medications and adjust accordingly"
        )

    return {
        "prescription_plan": state.prescription_plan,
        "clinical_alert": state.clinical_alert
    }

def clinical_lifestyle_advice(state: AgentState) -> dict:
    """  
        Gives a personalized, evidence-based lifestyle advice and metabolic optimization plan.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    prescription_plan = state.prescription_plan
    
    # Derive metabolic indicators
    bmi = patient.bmi
    pulse_pressure = patient.pulse_pressure
    
    # Basal Metabolic Rate (Mifflin-St Jeor Approximation)
    if patient.gender.lower() == "male":
        bmr = (10 * patient.weight) + (6.25 * patient.height * 100) - (5 * patient.age) + 5
    else:
        bmr = (10 * patient.weight) + (6.25 * patient.height * 100) - (5 * patient.age) - 161
        
    # Activity factor approximation
    if patient.avg_daily_steps < 5000:
        activity_factor = 1.2
    elif patient.avg_daily_steps < 8000:
        activity_factor = 1.375
    else:
        activity_factor = 1.55
        
    # Estimated TDEE (Total Daily Energy Expenditure)
    estimated_tdee = round(bmr * activity_factor, 2) 
    
    # Risk context formatting
    formatted_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score*100,1)}%"
        for risk in risk_assessment.disease_risks
    )
    
    medications_list = ", ".join(
        medication.name for medication in prescription_plan.medications
    ) if prescription_plan else None
    
    # Build structured lifestyle advice prompt
    lifestyle_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            STRICT RULES:
            - Use patient biometrics and risk scores.
            - Integrate medical evidence summary.
            - Do NOT diagnose.
            - Do NOT contradict prescribed medications.
            - Include intensity ranges and safety precautions.
            - Calibrate exercise to cardiovascular risk.
            - Calibrate diet to metabolic and lipid profile.
            - Address sleep physiology and metabolic regulation.
            - Provide clinically cautious recommendations.
            - Output strictly in structured JSON matching LifestylePlan schema.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
                Patient Profile:
                Age: {age}
                Gender: {gender}
                BMI: {bmi:.2f}
                Blood Pressure: {sbp}/{dbp}
                Pulse Pressure: {pulse_pressure}
                LDL: {ldl}
                HDL: {hdl}
                Triglycerides: {tg}
                Sugar Level: {sugar}
                Avg Sleep: {sleep} hrs
                Avg Daily Steps: {steps}
                Estimated TDEE: {tdee} kcal/day
                Current Medications: {medications}

                Disease Risks:
                {risks}

                Evidence Summary:
                {evidence}

                Generate a world-class, premium clinical lifestyle optimization plan.
                - Structured exercise plan: Include specific modalities (aerobic, resistance, flexibility), intensity (HRR%, RPE), duration, and frequency. Calibrate strictly to {risks}.
                - Structured dietary plan: Provide specific macronutrient ratios, micronutrient focus (e.g., sodium, potassium, fiber), and meal timing strategies based on {tdee} and metabolic profile.
                - Structured sleep optimization: Address circadian alignment, sleep hygiene, and physiological recovery based on current {sleep} hrs.
                - Structured metabolic optimization advice: Provide advanced strategies for glucose management, lipid optimization, and hormonal balance.
                
                Recommendations must be state-of-the-art, medically rigorous, and highly personalized. Never use generic or redundant dummy values.
                Never return NULL or empty values for any of the above fields.
                Ensure output strictly matches the LifestylePlan schema.
            """
        )
    ])
    
    lifestyle_chain = lifestyle_prompt | llm.with_structured_output(LifestylePlan)
    
    lifestyle_suggestions = lifestyle_chain.invoke({
        "age": patient.age,
        "gender": patient.gender,
        "bmi": bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "pulse_pressure": pulse_pressure,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "tg": patient.triglycerides,
        "sugar": patient.sugar_level,
        "sleep": patient.avg_sleep_hours,
        "steps": patient.avg_daily_steps,
        "tdee": estimated_tdee,
        "medications": medications_list,
        "risks": formatted_risks,
        "evidence": medical_evidence.clinical_summary
    })
    
    # Ensure all recommendations are present and premium
    if not lifestyle_suggestions.exercises:
        lifestyle_suggestions.exercises = ["Initiate Zone 2 cardiovascular training: 30-45 min, 3-4x weekly", "Incorporate progressive resistance training: 2x weekly, major muscle groups"]
    if not lifestyle_suggestions.diet:
        lifestyle_suggestions.diet = ["Adopt Mediterranean-style dietary pattern: focus on monounsaturated fats and high-fiber plant-based foods", "Limit processed carbohydrates and added sugars to optimize glycemic response"]
    if not lifestyle_suggestions.sleep:
        lifestyle_suggestions.sleep = ["Standardize sleep-wake cycle: ±30 min consistency", "Implement 60-min pre-sleep physiological down-regulation (no blue light, temperature optimization)"]
    if not lifestyle_suggestions.metabolic_advice:
        lifestyle_suggestions.metabolic_advice = ["Perform post-prandial walking: 10-15 min after largest meals to blunt glucose spikes", "Monitor continuous glucose trends or fasting levels weekly"]
    
    # Avoid high protein suggestions if kidney disease is present in history
    if any("kidney" in condition.lower() for condition in patient.medical_history):
        lifestyle_suggestions.diet = [
            diet for diet in lifestyle_suggestions.diet if "high_protein" not in diet.lower()
        ]
    
    # Deduplicate entries
    lifestyle_suggestions.exercises = list(dict.fromkeys(lifestyle_suggestions.exercises))
    lifestyle_suggestions.diet = list(dict.fromkeys(lifestyle_suggestions.diet))
    lifestyle_suggestions.sleep = list(dict.fromkeys(lifestyle_suggestions.sleep))
    lifestyle_suggestions.metabolic_advice = list(dict.fromkeys(lifestyle_suggestions.metabolic_advice))
    
    return {"lifestyle_plan": lifestyle_suggestions}

def clinical_strategy_synthesis(state: AgentState) -> dict:
    """
        Synthesizes a cohesive Clinical Road Map based on current treatment state.
        Provides a high-level, intuitive treatment strategy that makes sense for the patient.
    """
    patient = state.patient_profile
    risks = state.risk_assessment
    prescriptions = state.prescription_plan
    lifestyle = state.lifestyle_plan
    evidence = state.medical_evidence
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a senior clinical strategist. 
            Your goal is to synthesize all findings into a unified, logically intuitive "Clinical Road Map".
            
            Guidelines:
            - Provide a cohesive treatment narrative (Why this treatment? Why now?).
            - Prioritize the most critical interventions.
            - Explain the synergy between medications and lifestyle optimizations.
            - Set realistic expectations for monitoring and follow-up.
            - Use professional, optimistic, and action-oriented clinical language.
            - Do NOT diagnose.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Overview:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            Blood Pressure: {sbp}/{dbp}
            Sugar: {sugar} mmol/L
            
            Risk Summary:
            {risk_summary}
            
            Prescribed Medications:
            {medications}
            
            Lifestyle Strategy:
            - Exercise: {exercises}
            - Diet: {diet}
            
            Clinical Evidence Insight:
            {evidence_summary}
            
            Generate a world-class "Clinical Road Map" (2-3 paragraphs) that ties everything together into a strategic treatment plan.
            Do NOT use any em-dash or en-dash at all and NO NEED for any subheadings at all. Only return the content in paragraphs.
            """
        )
    ])
    
    medications_text = "\n".join([f"- {m.name}: {m.dose} ({m.notes})" for m in prescriptions.medications])
    
    road_map = llm.invoke(
        synthesis_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            sugar=patient.sugar_level,
            risk_summary=risks.risk_summary,
            medications=medications_text if medications_text else "None",
            exercises=", ".join(lifestyle.exercises[:2]),
            diet=", ".join(lifestyle.diet[:2]),
            evidence_summary=evidence.clinical_summary
        )
    ).content
    
    return {"treatment_road_map": road_map}

def alert_clinician(state: AgentState) -> dict:
    """ 
        Implements an alerting mechanism to notify clinicians of any significant changes in patient risk state.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    
    disease_risks = risk_assessment.disease_risks
    risk_flags = risk_assessment.risk_flags
    
    HIGH_THRESHOLD = 0.60
    CRITICAL_THRESHOLD = 0.80
    
    high_risk_conditions = []
    critical_conditions = []
    
    # Risk stratification
    for risk in disease_risks:
        if risk.risk_score >= CRITICAL_THRESHOLD:
            critical_conditions.append((risk.disease_name, risk.risk_score))
        elif risk.risk_score >= HIGH_THRESHOLD:
            high_risk_conditions.append((risk.disease_name, risk.risk_score))
            
    # Check for multi-morbidity (Meaning more than one condition is at high or critical risk)
    multi_morbidity_flag = False
    
    if len(critical_conditions) > 1:
        multi_morbidity_flag = True
    
    # Escalation due to interacting risks
    interaction_flags = []
    
    if "high_diabetes_risk" in risk_flags and "high_cardiovascular_risk" in risk_flags:
        interaction_flags.append("Diabetes-Cardiovascular interaction risk")
    
    if "metabolic_syndrome_risk" in risk_flags and "hypertension_risk" in risk_flags:
        interaction_flags.append("Metabolic syndrome-Hypertension interaction risk")
        
    if interaction_flags:
        multi_morbidity_flag = True
    
    # Assign emergency tiers
    if len(critical_conditions) >= 2:
        urgency = "CRITICAL"
    elif critical_conditions:
        urgency = "HIGH"
    elif multi_morbidity_flag:
        urgency = "HIGH"
    elif high_risk_conditions:
        urgency = "MODERATE"
    else:
        urgency = "LOW"
    
    # If urgency is LOW, no alert escalation is needed
    if urgency == "LOW":
        return {
            "clinical_alert": ClinicalAlert(
                urgency="LOW",
                message="No immediate clinician escalation required.",
                conditions_flagged=[],
                interaction_flags=[],
                recommended_action="Continue routine monitoring."
            )
        }
        
    # Create structured clinical context
    formatted_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score * 100, 1)}%"
        for risk in disease_risks
    )

    critical_text = "\n".join(
        f"{disease} ({round(score * 100, 1)}%)"
        for disease, score in critical_conditions
    ) if critical_conditions else "None"

    high_risk_text = "\n".join(
        f"{disease} ({round(score * 100, 1)}%)"
        for disease, score in high_risk_conditions
    ) if high_risk_conditions else "None"
    
    interaction_text = "\n".join(interaction_flags) if interaction_flags else "None"
    
    # Create a structured clinician alert prompt
    alert_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a hospital-grade clinical escalation engine.

            Your role:
            - Generate a structured clinician alert.
            - Use cautious, non-diagnostic language.
            - Highlight urgency tier.
            - Identify potential complications.
            - Suggest monitoring intensity or referral level.
            - DO NOT prescribe medications.
            - DO NOT diagnose.
            - Maintain clinical professionalism.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Profile:
            Age: {age}
            Gender: {gender}
            BMI: {bmi:.2f}
            Blood Pressure: {sbp}/{dbp}
            LDL: {ldl}
            HDL: {hdl}
            Triglycerides: {tg}
            Sugar Level: {sugar}

            Risk Scores:
            {risks}

            Critical Conditions:
            {critical}

            High Risk Conditions:
            {high}

            Interaction Risks:
            {interaction}

            Evidence Context:
            {evidence}

            Urgency Tier: {urgency}

            Generate a concise but clinically rigorous alert summary.
            """
        )
    ])
    
    alert_summary = llm.invoke(
        alert_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            ldl=patient.ldl_cholesterol,
            hdl=patient.hdl_cholesterol,
            tg=patient.triglycerides,
            sugar=patient.sugar_level,
            risks=formatted_risks,
            critical=critical_text,
            high=high_risk_text,
            interaction=interaction_text,
            evidence=medical_evidence.clinical_summary if medical_evidence else "No evidence found",
            urgency=urgency
        )
    ).content
    
    # Recommend escalation actions
    if urgency == "CRITICAL":
        recommended_action = (
            "Initiate immediate hospitalization and refer to a critical care unit. "
            "Conduct urgent cardiovascular and metabolic panel reassessments."
        )
    elif urgency == "HIGH":
        recommended_action = (
            "Schedule priority clinical evaluation within 1-2 weeks. "
            "Increase monitoring intensity."
        )
    else:
        recommended_action = (
            "Monitor closely and reassess clinical conditions within 4-6 weeks."
        )
        
    clinical_alert = ClinicalAlert(
        urgency=urgency,
        message=alert_summary,
        conditions_flagged=[d for d, _ in critical_conditions + high_risk_conditions],
        interaction_flags=interaction_flags,
        recommended_action=recommended_action
    )
    
    return {"clinical_alert": clinical_alert}

def general_wellness_synthesis(state: AgentState) -> dict:
    """
    Provides high-level preventative health and wellness advice for low-risk patients.
    """
    patient = state.patient_profile
    
    wellness_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a preventative health specialist. 
            The patient is currently in a low-risk category.
            Your goal is to provide a positive, encouraging "Wellness Road Map" focused on longevity and health optimization.
            
            Guidelines:
            - Focus on preventative measures (Sleep, Diet, Stress).
            - Set optimization goals (e.g., step targets, sleep hygiene).
            - Keep it professional, optimistic, and easy to follow.
            - Do NOT provide clinical prescriptions or intensive medical warnings.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Overview:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            
            Current Health Markers:
            BP: {sbp}/{dbp}, Sugar: {sugar} mmol/L
            
            Generate a concise "Wellness Road Map" (1-2 paragraphs) for this patient.
            """
        )
    ])
    
    wellness_map = llm.invoke(
        wellness_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            sugar=patient.sugar_level
        )
    ).content
    
    return {"treatment_road_map": wellness_map}


# Initiate a StateGraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("collect_patient_data", collect_patient_data)
graph.add_node("detect_early_disease", early_disease_detection)
graph.add_node("alert_clinician", alert_clinician)
graph.add_node("fetch_medical_literature", fetch_medical_literature)
graph.add_node("prescribe_medications", prescribe_medications)
graph.add_node("drug_safety_guardrails", drug_safety_guardrail)
graph.add_node("give_lifestyle_advice", clinical_lifestyle_advice)
graph.add_node("clinical_strategy_synthesis", clinical_strategy_synthesis)
graph.add_node("general_wellness_synthesis", general_wellness_synthesis)

# Add edges to the graph
graph.add_edge(START, "collect_patient_data")
graph.add_edge("collect_patient_data", "detect_early_disease")

# Conditional Risk Triage
graph.add_conditional_edges(
    "detect_early_disease",
    clinical_triage_router,
    {
        "high_risk": "alert_clinician",
        "low_risk": "general_wellness_synthesis"
    }
)

# High-Risk Path
graph.add_edge("alert_clinician", "fetch_medical_literature")

# Parallel Execution Track (Medications & Lifestyle)
graph.add_edge("fetch_medical_literature", "prescribe_medications")
graph.add_edge("fetch_medical_literature", "give_lifestyle_advice")

# Add guardrails check immediately after prescriptions are generated
graph.add_edge("prescribe_medications", "drug_safety_guardrails")

# Synchronizing parallel tracks into the final clinical synthesis
graph.add_edge("drug_safety_guardrails", "clinical_strategy_synthesis")
graph.add_edge("give_lifestyle_advice", "clinical_strategy_synthesis")

graph.add_edge("clinical_strategy_synthesis", END)
graph.add_edge("general_wellness_synthesis", END)

# Compile the graph
workflow = graph.compile()

# Visualize the graph
try:
    with open("clinical_patient_monitoring_workflow.png", "wb") as f:
        f.write(workflow.get_graph().draw_mermaid_png())
    print("Workflow visualization saved to 'clinical_patient_monitoring_workflow.png'")
except Exception as e:
    print(f"Could not save workflow visualization: {e}")