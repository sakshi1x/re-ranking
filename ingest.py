import logging
import chromadb
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

def populate_sample_data(
    directory: str = "./chroma_db",
    collection_name: str = "generic_collection",
    embedding_model: str = "nomic-embed-text",
    embedding_url: str = ""
):
    """
    Populate a ChromaDB collection with 100 generic sample documents from various domains.

    Args:
        directory (str): Directory path for ChromaDB persistence.
        collection_name (str): Name of the collection to create or overwrite.
        embedding_model (str): Embedding model name for OllamaEmbeddings.
        embedding_url (str): Base URL for the embedding service.
    """
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_url)
    client = chromadb.PersistentClient(path=directory)

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"No existing collection to delete: Collection {collection_name} does not exist.")

    collection = client.create_collection(name=collection_name)

    # 100 generic sample documents organized by domain
    sample_documents = [
        # Technology (20 documents)
        {"id": "1", "content": "Cloud computing enhances scalability for modern enterprises.", "metadata": {"source": "Technology"}},
        {"id": "2", "content": "Cybersecurity frameworks protect sensitive data from breaches.", "metadata": {"source": "Technology"}},
        {"id": "3", "content": "Quantum computing promises breakthroughs in complex problem-solving.", "metadata": {"source": "Technology"}},
        {"id": "4", "content": "Blockchain ensures secure and transparent transactions.", "metadata": {"source": "Technology"}},
        {"id": "5", "content": "5G networks enable faster and more reliable data transmission.", "metadata": {"source": "Technology"}},
        {"id": "6", "content": "Artificial intelligence automates repetitive tasks efficiently.", "metadata": {"source": "Technology"}},
        {"id": "7", "content": "Internet of Things connects devices for smarter homes.", "metadata": {"source": "Technology"}},
        {"id": "8", "content": "Virtual reality enhances immersive gaming experiences.", "metadata": {"source": "Technology"}},
        {"id": "9", "content": "Augmented reality transforms product visualization in retail.", "metadata": {"source": "Technology"}},
        {"id": "10", "content": "Robotics revolutionizes automation in manufacturing.", "metadata": {"source": "Technology"}},
        {"id": "11", "content": "Machine learning optimizes predictive analytics.", "metadata": {"source": "Technology"}},
        {"id": "12", "content": "Big data platforms process vast amounts of information.", "metadata": {"source": "Technology"}},
        {"id": "13", "content": "Smart grids improve energy distribution efficiency.", "metadata": {"source": "Technology"}},
        {"id": "14", "content": "Drones enhance delivery systems in remote areas.", "metadata": {"source": "Technology"}},
        {"id": "15", "content": "Wearable devices track fitness and health metrics.", "metadata": {"source": "Technology"}},
        {"id": "16", "content": "Smart cities leverage IoT for urban development.", "metadata": {"source": "Technology"}},
        {"id": "17", "content": "Open-source software fosters collaborative innovation.", "metadata": {"source": "Technology"}},
        {"id": "18", "content": "Data encryption secures online communications.", "metadata": {"source": "Technology"}},
        {"id": "19", "content": "Edge computing reduces latency in real-time applications.", "metadata": {"source": "Technology"}},
        {"id": "20", "content": "DevOps practices streamline software development cycles.", "metadata": {"source": "Technology"}},

        # Healthcare (20 documents)
        {"id": "21", "content": "Telemedicine improves healthcare access in rural regions.", "metadata": {"source": "Healthcare"}},
        {"id": "22", "content": "Personalized medicine tailors treatments to individual patients.", "metadata": {"source": "Healthcare"}},
        {"id": "23", "content": "Vaccination programs reduce the spread of infectious diseases.", "metadata": {"source": "Healthcare"}},
        {"id": "24", "content": "Gene editing technologies advance medical research.", "metadata": {"source": "Healthcare"}},
        {"id": "25", "content": "Mental health apps support emotional well-being.", "metadata": {"source": "Healthcare"}},
        {"id": "26", "content": "Robotic surgery enhances precision in complex operations.", "metadata": {"source": "Healthcare"}},
        {"id": "27", "content": "Electronic health records streamline patient data management.", "metadata": {"source": "Healthcare"}},
        {"id": "28", "content": "Wearable health monitors detect early signs of illness.", "metadata": {"source": "Healthcare"}},
        {"id": "29", "content": "AI diagnostics predict diseases with high accuracy.", "metadata": {"source": "Healthcare"}},
        {"id": "30", "content": "Public health campaigns promote preventive care.", "metadata": {"source": "Healthcare"}},
        {"id": "31", "content": "Genomics drives advancements in personalized treatments.", "metadata": {"source": "Healthcare"}},
        {"id": "32", "content": "Telehealth counseling supports mental health remotely.", "metadata": {"source": "Healthcare"}},
        {"id": "33", "content": "Nutrition plans aid in patient recovery and wellness.", "metadata": {"source": "Healthcare"}},
        {"id": "34", "content": "Medical imaging improves diagnostic precision.", "metadata": {"source": "Healthcare"}},
        {"id": "35", "content": "Healthcare analytics optimize hospital operations.", "metadata": {"source": "Healthcare"}},
        {"id": "36", "content": "Chronic disease management enhances quality of life.", "metadata": {"source": "Healthcare"}},
        {"id": "37", "content": "Pharmaceutical research develops life-saving drugs.", "metadata": {"source": "Healthcare"}},
        {"id": "38", "content": "Health education empowers patients to make informed choices.", "metadata": {"source": "Healthcare"}},
        {"id": "39", "content": "Rehabilitation technologies aid physical recovery.", "metadata": {"source": "Healthcare"}},
        {"id": "40", "content": "Epidemiology tracks and controls disease outbreaks.", "metadata": {"source": "Healthcare"}},

        # Business (20 documents)
        {"id": "41", "content": "Effective customer retention strategies boost business growth.", "metadata": {"source": "Business"}},
        {"id": "42", "content": "AI-driven analytics optimize supply chain management.", "metadata": {"source": "Business"}},
        {"id": "43", "content": "Digital marketing increases brand visibility online.", "metadata": {"source": "Business"}},
        {"id": "44", "content": "Remote work tools enhance team collaboration.", "metadata": {"source": "Business"}},
        {"id": "45", "content": "E-commerce platforms drive retail innovation.", "metadata": {"source": "Business"}},
        {"id": "46", "content": "Corporate social responsibility builds consumer trust.", "metadata": {"source": "Business"}},
        {"id": "47", "content": "SEO strategies improve website search rankings.", "metadata": {"source": "Business"}},
        {"id": "48", "content": "Supply chain transparency enhances customer loyalty.", "metadata": {"source": "Business"}},
        {"id": "49", "content": "Employee training programs increase productivity.", "metadata": {"source": "Business"}},
        {"id": "50", "content": "Business intelligence tools analyze market trends.", "metadata": {"source": "Business"}},
        {"id": "51", "content": "Agile methodologies improve project adaptability.", "metadata": {"source": "Business"}},
        {"id": "52", "content": "Financial forecasting predicts revenue growth.", "metadata": {"source": "Business"}},
        {"id": "53", "content": "CRM systems strengthen customer relationships.", "metadata": {"source": "Business"}},
        {"id": "54", "content": "Lean practices reduce operational waste.", "metadata": {"source": "Business"}},
        {"id": "55", "content": "Competitor analysis informs strategic planning.", "metadata": {"source": "Business"}},
        {"id": "56", "content": "Brand loyalty programs encourage repeat purchases.", "metadata": {"source": "Business"}},
        {"id": "57", "content": "Risk management mitigates business uncertainties.", "metadata": {"source": "Business"}},
        {"id": "58", "content": "Pricing strategies enhance market competitiveness.", "metadata": {"source": "Business"}},
        {"id": "59", "content": "Digital transformation modernizes business operations.", "metadata": {"source": "Business"}},
        {"id": "60", "content": "Customer feedback loops refine product offerings.", "metadata": {"source": "Business"}},

        # Education (15 documents)
        {"id": "61", "content": "Online learning platforms revolutionize education delivery.", "metadata": {"source": "Education"}},
        {"id": "62", "content": "Virtual reality enhances immersive learning experiences.", "metadata": {"source": "Education"}},
        {"id": "63", "content": "Gamification improves student engagement in classrooms.", "metadata": {"source": "Education"}},
        {"id": "64", "content": "Adaptive learning adjusts to individual student needs.", "metadata": {"source": "Education"}},
        {"id": "65", "content": "STEM education fosters innovation in young minds.", "metadata": {"source": "Education"}},
        {"id": "66", "content": "Blended learning combines online and in-person teaching.", "metadata": {"source": "Education"}},
        {"id": "67", "content": "MOOCs provide free access to university courses.", "metadata": {"source": "Education"}},
        {"id": "68", "content": "Teacher training improves classroom effectiveness.", "metadata": {"source": "Education"}},
        {"id": "69", "content": "EdTech tools transform traditional education systems.", "metadata": {"source": "Education"}},
        {"id": "70", "content": "Student assessments measure academic progress.", "metadata": {"source": "Education"}},
        {"id": "71", "content": "Distance learning supports remote education access.", "metadata": {"source": "Education"}},
        {"id": "72", "content": "Critical thinking skills enhance problem-solving abilities.", "metadata": {"source": "Education"}},
        {"id": "73", "content": "Vocational training prepares students for practical jobs.", "metadata": {"source": "Education"}},
        {"id": "74", "content": "Educational research drives teaching innovations.", "metadata": {"source": "Education"}},
        {"id": "75", "content": "Scholarships make higher education more accessible.", "metadata": {"source": "Education"}},

        # Environment/Agriculture (10 documents)
        {"id": "76", "content": "Sustainable farming practices reduce environmental impact.", "metadata": {"source": "Agriculture"}},
        {"id": "77", "content": "Renewable energy sources combat climate change.", "metadata": {"source": "Environment"}},
        {"id": "78", "content": "Eco-friendly packaging minimizes retail waste.", "metadata": {"source": "Environment"}},
        {"id": "79", "content": "Precision agriculture increases crop yields sustainably.", "metadata": {"source": "Agriculture"}},
        {"id": "80", "content": "Solar energy adoption reduces reliance on fossil fuels.", "metadata": {"source": "Environment"}},
        {"id": "81", "content": "Organic farming meets growing consumer demand.", "metadata": {"source": "Agriculture"}},
        {"id": "82", "content": "Recycling programs lower environmental pollution.", "metadata": {"source": "Environment"}},
        {"id": "83", "content": "Water conservation techniques support sustainable living.", "metadata": {"source": "Environment"}},
        {"id": "84", "content": "Biodiversity efforts protect endangered species.", "metadata": {"source": "Environment"}},
        {"id": "85", "content": "Urban farming promotes local food production.", "metadata": {"source": "Agriculture"}},

        # Finance/Marketing (10 documents)
        {"id": "86", "content": "Automated trading systems improve market efficiency.", "metadata": {"source": "Finance"}},
        {"id": "87", "content": "Financial literacy programs empower personal budgeting.", "metadata": {"source": "Finance"}},
        {"id": "88", "content": "Crowdfunding platforms support startup innovation.", "metadata": {"source": "Finance"}},
        {"id": "89", "content": "Cryptocurrency reshapes global financial systems.", "metadata": {"source": "Finance"}},
        {"id": "90", "content": "Investment apps democratize wealth management.", "metadata": {"source": "Finance"}},
        {"id": "91", "content": "Social media marketing engages modern audiences.", "metadata": {"source": "Marketing"}},
        {"id": "92", "content": "Influencer campaigns boost product awareness.", "metadata": {"source": "Marketing"}},
        {"id": "93", "content": "Content marketing builds long-term customer trust.", "metadata": {"source": "Marketing"}},
        {"id": "94", "content": "Email campaigns drive personalized customer outreach.", "metadata": {"source": "Marketing"}},
        {"id": "95", "content": "Market research identifies consumer trends accurately.", "metadata": {"source": "Marketing"}},

        # Miscellaneous (5 documents)
        {"id": "96", "content": "Data privacy laws regulate corporate data usage.", "metadata": {"source": "Legal"}},
        {"id": "97", "content": "Space exploration advances scientific discovery.", "metadata": {"source": "Science"}},
        {"id": "98", "content": "Tourism boosts local economies through travel.", "metadata": {"source": "Business"}},
        {"id": "99", "content": "Ethical AI ensures fairness in automated decisions.", "metadata": {"source": "Technology"}},
        {"id": "100", "content": "Smart home devices improve energy efficiency.", "metadata": {"source": "Technology"}},
    ]

    # Ensure exactly 100 documents
    assert len(sample_documents) == 100, f"Expected 100 documents, got {len(sample_documents)}"

    documents = [doc["content"] for doc in sample_documents]
    embeddings_list = embeddings.embed_documents(documents)
    metadatas = [{"source": doc["metadata"]["source"], "id": doc["id"]} for doc in sample_documents]
    ids = [doc["id"] for doc in sample_documents]

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Successfully populated {len(sample_documents)} documents into collection.")

# if __name__ == "__main__":
#     # Example usage
#     logging.basicConfig(level=logging.INFO)
#     populate_sample_data(
#         directory="./chroma_db",
#         collection_name="generic_collection",
#         embedding_url="https://jo3m4y06rnnwhaz.askbhunte.com"  # Adjust as needed
#     )