# RentRakshak - Module 1: UML & Architecture Diagrams

This document contains text-based UML and architecture diagrams for the RentRakshak Module 1 (Pre-Visit Guard) project, adhering to the SOLID principles outlined in the Low-Level Design (LLD).

## 1. System Architecture (Component Diagram)
This diagram illustrates the flow of data from the Next.js frontend through the FastAPI backend, the ML analyzers, and finally to the local data stores.

```mermaid
graph TD
    %% Frontend Components
    subgraph Frontend [Next.js / React Frontend]
        UI_Input(InputController<br/>URL / Screenshot)
        UI_Score(ScoreDial<br/>0-100 Gauge)
        UI_Proof(ProofCards<br/>Dynamic Explanations)
        UI_Feedback(FeedbackForm<br/>Active Learning)
    end

    %% Backend Services
    subgraph Backend [FastAPI Backend]
        API_Main[main.py Route Handlers]
        API_Scrape[ListingScraper]
        API_Score[EnsembleScorer]
        API_LLM[Ollama Local LLM<br/>phi3:mini]
    end

    %% ML Analyzers
    subgraph ML_Services [ML Risk Analyzers]
        ML_Vision[VisionAnomalyAnalyzer<br/>pHash, CLIP, MobileNet]
        ML_Price[PriceAnomalyAnalyzer<br/>RandomForest]
        ML_Text[TextScamAnalyzer<br/>IndicBERT ONNX]
    end

    %% Databases
    subgraph Data [Data & Storage Layer]
        DB_Duck[(DuckDB<br/>Image pHashes)]
        DB_Qdrant[(Qdrant Docker<br/>CLIP Embeddings)]
        DB_SQLite[(SQLite<br/>Feedback & Logs)]
    end

    %% Data Flow
    UI_Input -->|POST /check-listing| API_Main
    API_Main --> API_Scrape
    API_Scrape --> API_Score
    
    API_Score -->|Image Bytes| ML_Vision
    API_Score -->|Price/Amenities| ML_Price
    API_Score -->|Description Text| ML_Text
    
    ML_Vision -->|Hash Search| DB_Duck
    ML_Vision -->|Vector Search| DB_Qdrant
    
    ML_Vision -->|Image Score| API_Score
    ML_Price -->|Price Score| API_Score
    ML_Text -->|Text Score| API_Score
    
    API_Score -->|Combined Risks| API_LLM
    API_LLM -->|Generate Explanations| API_Main
    
    API_Main -->|JSON Response| UI_Score
    API_Main -->|JSON Response| UI_Proof
    
    UI_Feedback -->|POST /feedback| DB_SQLite
```

## 2. SOLID Class Diagram
This class diagram highlights the core Object-Oriented structure of the backend. It demonstrates how interfaces (`IRiskAnalyzer`, `IImageProcessor`) are used to decouple dependencies, fulfilling the Open/Closed and Dependency Inversion principles.

```mermaid
classDiagram
    direction TB

    %% Core Interfaces
    class IRiskAnalyzer {
        <<Interface>>
        +analyze(listing_data: dict) tuple~float, str~
    }

    class IFeedbackRepository {
        <<Interface>>
        +save_user_feedback(feedback_data: dict) bool
    }

    class IImageProcessor {
        <<Interface>>
        +generate_phash(image_bytes: bytes) str
        +get_real_room_probability(image_bytes: bytes) float
    }

    class ITextProcessor {
        <<Interface>>
        +detect_pressure_language(text: str) float
    }

    %% Concrete Implementations (Analyzers)
    class VisionAnomalyAnalyzer {
        -image_processor: IImageProcessor
        +analyze(listing_data: dict) tuple~float, str~
    }

    class PriceAnomalyAnalyzer {
        -feature_engineer: PriceFeatureEngineer
        +analyze(listing_data: dict) tuple~float, str~
    }

    class TextScamAnalyzer {
        -text_processor: ITextProcessor
        +analyze(listing_data: dict) tuple~float, str~
    }

    %% Concrete Implementations (Data)
    class SQLiteFeedbackRepository {
        -db_connection: Connection
        +save_user_feedback(feedback_data: dict) bool
    }

    %% Utilities
    class ListingScraper {
        +extract_listing_data(url: str) dict
    }

    class PriceFeatureEngineer {
        +engineer_features(raw_data: dict) list
    }

    %% Orchestrator
    class EnsembleScorer {
        -analyzers: List~IRiskAnalyzer~
        +calculate_final_score(listing_data: dict) dict
    }

    %% Relationships (Inheritance/Realization)
    IRiskAnalyzer <|-- VisionAnomalyAnalyzer : Implements
    IRiskAnalyzer <|-- PriceAnomalyAnalyzer : Implements
    IRiskAnalyzer <|-- TextScamAnalyzer : Implements
    
    IFeedbackRepository <|-- SQLiteFeedbackRepository : Implements

    %% Relationships (Dependencies & Aggregations)
    VisionAnomalyAnalyzer --> IImageProcessor : Uses
    PriceAnomalyAnalyzer --> PriceFeatureEngineer : Uses
    TextScamAnalyzer --> ITextProcessor : Uses
    
    EnsembleScorer o-- IRiskAnalyzer : Aggregates
```