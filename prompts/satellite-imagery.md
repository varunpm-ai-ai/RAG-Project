You are now building PHASE 2 of our college mini-project.

PHASE 1 has already produced:

1. A trained DOTA object-detection model.
2. An inference-ready best model artifact.
3. A reusable predictor/inference function.
4. A canonical 15-class mapping.
5. A synthetic/demo knowledge document containing information for all 15 classes.

Your task is to build a polished but simple Streamlit application that integrates the object detector with a RAG pipeline.

This is a COLLEGE MVP.

Do NOT over-engineer it.

The entire application should preferably run as a single Python/Streamlit application.

Do NOT create FastAPI unless there is a genuine technical reason that Streamlit cannot reasonably handle the requirement.

No Kubernetes.
No Kafka.
No RabbitMQ.
No microservices.
No authentication system.
No distributed architecture.
No cloud infrastructure.
No unnecessary database.

The objective is to demonstrate:

IMAGE
→ OBJECT DETECTION
→ CATEGORY
→ CATEGORY-SPECIFIC RAG
→ ANSWER

and then:

UPLOAD NEW DOCUMENT
→ UPDATE KNOWLEDGE BASE
→ SAME QUESTION
→ DIFFERENT ANSWER BASED ON NEW DOCUMENT

---

# 1. CORE USER EXPERIENCE

The application should have two main upload areas.

### Upload A — Image

The user uploads an aerial image.

The application should:

1. Display the uploaded image.
2. Run the trained DOTA detector.
3. Display the annotated image.
4. Display all detected objects.
5. Display their confidence scores.
6. Identify the detected category/categories.
7. Use the detected category to control the RAG retrieval.

Example:

Detected:

Plane — 94.7%

Small Vehicle — 88.2%

Small Vehicle — 81.4%

Do NOT assume an image contains only one object.

---

# 2. IMPORTANT RAG BEHAVIOR

The detected class must influence document retrieval.

For example:

Image detector returns:

plane

The RAG system should retrieve information associated with:

plane

It should NOT blindly retrieve unrelated documents about:

ship
bridge
vehicle
harbor

unless the user explicitly asks for broader information.

The architecture should conceptually be:

Image
↓
Detector
↓
Detected category
↓
Category filter
↓
Relevant document chunks
↓
LLM
↓
Answer

---

# 3. BASE KNOWLEDGE DOCUMENT

PHASE 1 created a synthetic base document containing information about all 15 DOTA classes.

Load this document automatically when the application starts.

The application should display something like:

Knowledge Base

Base Knowledge: ✓ Loaded

Categories Available: 15

Do not require the user to manually upload the base document every time.

The base document is the initial knowledge source.

---

# 4. DOCUMENT UPLOAD

Create a second upload component specifically for adding new knowledge.

Label it clearly, for example:

"Add Knowledge Document"

Support at minimum:

* JSON
* TXT

If practical, also support:

* PDF
* DOCX

But JSON/TXT functionality is mandatory for the MVP.

The user should be able to upload approximately five additional documents during the demonstration.

The application should:

1. Read the document.
2. Extract text.
3. Determine/accept the relevant DOTA category.
4. Split the text into chunks.
5. Generate embeddings.
6. Store the chunks in a vector store.
7. Attach metadata.

Metadata should contain at least:

* category
* source
* document name
* document type

Example:

{
"category": "plane",
"source": "uploaded_document_1.json"
}

---

# 5. CATEGORY SELECTION FOR UPLOADED DOCUMENTS

Do NOT rely entirely on an LLM to guess the category.

For the MVP, provide a simple UI control:

"Document Category"

with the 15 DOTA categories.

Example:

Document Category:
[ Plane ▼ ]

This makes the demo deterministic and easier for faculty to understand.

The user can upload:

plane_information.json

and select:

Plane

Then the document becomes associated with Plane.

If multiple categories are genuinely needed, allow an optional multi-select, but keep the default workflow simple.

---

# 6. VECTOR DATABASE

Use a lightweight local vector store.

Prefer:

FAISS

or:

Chroma

Choose whichever is simpler and more reliable in the environment.

Do NOT introduce a remote vector database.

Persist the vector index locally so that restarting Streamlit does not unnecessarily destroy the knowledge base.

A simple structure is enough:

rag_storage/

```
index/
metadata/
documents/
```

The exact implementation is up to you.

---

# 7. EMBEDDINGS

Use a practical embedding model.

Prefer a lightweight locally available sentence-transformer model if possible.

Do not use an enormous embedding model.

The application should:

document
→ chunks
→ embeddings
→ vector store

and during retrieval:

question
→ embedding
→ similarity search
→ relevant chunks

---

# 8. LLM

Create a clean abstraction for the LLM.

The application should be able to use an LLM through an environment variable/API key rather than hard-coding credentials.

For example:

.env

LLM_API_KEY=...

Do NOT commit secrets.

If a suitable local LLM is already available, support it where practical.

However, the primary requirement is simply:

retrieved context
+
user question
→
LLM
→
answer

Keep the LLM integration isolated in a module such as:

src/rag/llm.py

so it can easily be changed later.

---

# 9. RAG PROMPTING

The LLM must be instructed to answer using retrieved information.

The system prompt should communicate something conceptually similar to:

"You are answering questions about objects detected in aerial imagery.

Use the supplied retrieved documents as the primary knowledge source.

Do not invent facts that are not supported by the retrieved context.

If the retrieved documents do not contain enough information to answer the question, explicitly say that the available knowledge base does not contain enough information."

The detected category should also be included.

For example:

Detected Category:
Plane

Retrieved Context:
...

Question:
What information is available about this object?

This makes the relationship between computer vision and RAG explicit.

---

# 10. THE MOST IMPORTANT DEMO

The application MUST make this workflow very easy to demonstrate.

### STEP 1

Upload an image containing a Plane.

Detector:

Plane — 95%

### STEP 2

Ask:

"What information is available about this object?"

The application retrieves the base Plane information.

Display:

Answer based on:
Base Knowledge

### STEP 3

Upload a new document.

For example:

plane_additional_demo.json

Select:

Plane

The application indexes it.

### STEP 4

Ask THE SAME QUESTION again.

The answer should now incorporate the newly uploaded information.

Display something like:

Answer based on:

* Base Knowledge
* plane_additional_demo.json

This is the central proof that the RAG system is working.

---

# 11. DOCUMENT SOURCE DISPLAY

Make the retrieved sources visible.

Do not hide the RAG process completely.

Under the answer display:

Sources Used

1. base_knowledge.json
2. plane_additional_demo.json

Optionally provide expandable sections:

Source 1

> Retrieved text chunk...

Source 2

> Retrieved text chunk...

This is especially important for a college demonstration because faculty can visually understand that the answer came from the uploaded documents.

---

# 12. STREAMLIT UI DESIGN

The UI should look polished and intuitive rather than like a default Streamlit prototype.

Design it specifically for faculty members who may not be technically strong.

Use:

* clear section headings
* cards/containers
* clean spacing
* concise labels
* icons where appropriate
* status indicators
* confidence badges
* expandable technical details
* responsive layout

Avoid excessive animations or unnecessary visual complexity.

Suggested overall layout:

---

DOTA Vision + Knowledge Assistant

"Detect an aerial object and explore its associated knowledge."

---

LEFT / MAIN AREA

IMAGE DETECTION

[ Upload Aerial Image ]

[ image preview ]

[ Run Detection ]

Detection Results

┌──────────────────────────────┐
│ Plane                        │
│ Confidence: 95.2%            │
└──────────────────────────────┘

[ Annotated image ]

---

RAG / KNOWLEDGE AREA

Detected Category:
PLANE

Knowledge Sources:
✓ Base Knowledge
✓ 2 Uploaded Documents

---

Ask About The Detected Object

[ text input ]

[ Ask Question ]

---

ANSWER

[ generated answer ]

Sources Used
▼ Base Knowledge
▼ Additional Document

---

SIDEBAR

Knowledge Base

Base Knowledge:
✓ Loaded

Uploaded Documents:
3

[ Add Knowledge Document ]

Document Category:
[ Plane ▼ ]

[ Upload ]

---

# 13. TWO DISTINCT WORKFLOWS

Keep the UI conceptually divided into two workflows.

### Workflow A — Detect

Upload image.

Run detector.

See detected object.

### Workflow B — Expand Knowledge

Upload document.

Choose category.

Index document.

Ask questions.

This makes the application easy to understand.

---

# 14. MULTIPLE DETECTED CLASSES

The model can detect multiple categories in one image.

For example:

Plane
Small Vehicle
Bridge

The MVP should handle this gracefully.

Display all detections.

Then allow the user to choose which detected category they want to ask about.

Example:

Detected Categories:

[ Plane ] [ Small Vehicle ] [ Bridge ]

If the user selects:

Plane

then RAG retrieval should be filtered to Plane.

This is much better than arbitrarily selecting the first detection.

---

# 15. IF ONLY ONE CATEGORY IS DETECTED

Automatically select it.

Example:

Detected:

Plane — 94.3%

Then:

Selected Knowledge Category:
Plane

No additional interaction should be required.

---

# 16. CHAT / QUESTION INTERFACE

The user should be able to ask questions about the selected detected category.

For example:

"What is this object?"

"What information do we have about it?"

"What additional information was provided in the uploaded document?"

"Summarize the available information."

"Compare the information from the base document and uploaded document."

The RAG pipeline should retrieve relevant chunks rather than simply dumping the entire document into the LLM.

---

# 17. SESSION STATE

Use Streamlit session state appropriately.

Maintain:

* uploaded image
* detection results
* selected category
* uploaded document list
* current question
* answer
* retrieval results

Do not repeatedly reload/recompute the detector or embedding model unnecessarily on every Streamlit rerun.

Use caching mechanisms appropriately.

For example:

@st.cache_resource

for heavyweight models.

---

# 18. MODEL LOADING

Load the Phase 1 best model from the expected artifact location.

Do not duplicate the training pipeline.

Create one clean interface such as:

predictor.predict(image)

The Streamlit application should consume the already-trained model.

If the model cannot be found, display a clear UI error:

"Detection model not found. Please complete Phase 1 training first."

Do not silently fail.

---

# 19. PROJECT STRUCTURE

Create a clean structure similar to:

project/
│
├── app.py
│
├── src/
│   ├── detection/
│   │   └── predictor.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── pipeline.py
│   │
│   ├── documents/
│   │   └── loader.py
│   │
│   └── utils/
│
├── artifacts/
│   └── best_model/
│
├── knowledge_base/
│   └── base_knowledge.json
│
├── rag_storage/
│
├── uploaded_documents/
│
├── requirements.txt
├── .env.example
└── README.md

Adapt this structure if needed, but keep the project simple.

---

# 20. FASTAPI DECISION

Do NOT create FastAPI by default.

First implement everything directly in Streamlit.

Only introduce FastAPI if there is a concrete limitation that makes the Streamlit-only architecture impractical.

For this MVP, the preferred architecture is:

Streamlit
│
├── Object Detector
│
├── Document Loader
│
├── Embedding Model
│
├── Vector Store
│
└── LLM

This is sufficient.

---

# 21. DOCUMENT MANAGEMENT

Provide a small knowledge-base status section.

Show:

Base documents:
1

Uploaded documents:
3

Indexed chunks:
47

Current selected category:
Plane

Also provide a way to remove an uploaded document if practical.

If implementing deletion becomes unnecessarily complicated, prioritize upload/index/retrieval first.

---

# 22. RESET / CLEAR FUNCTION

Provide a simple:

"Clear Session"

or

"Reset Demo"

button.

It should clear temporary session state without accidentally deleting the base knowledge.

If a destructive action deletes indexed uploaded documents, ask for confirmation.

---

# 23. ERROR HANDLING

Handle:

* missing model
* invalid image
* unsupported document
* malformed JSON
* empty document
* missing API key
* LLM failure
* embedding failure
* vector-store failure
* no detected object
* no relevant retrieved context

Messages should be understandable to a non-technical user.

Avoid exposing raw Python stack traces in the main UI.

Technical errors can be available inside an expandable "Technical Details" section.

---

# 24. SECURITY

Even though this is only an MVP:

* never hard-code API keys
* use environment variables
* create .env.example
* add .env to .gitignore
* do not upload secrets to GitHub

Uploaded documents are local demo data.

---

# 25. PERFORMANCE

The application should not reload large models unnecessarily.

Cache:

* object detection model
* embedding model
* LLM client if appropriate
* vector store where safe

The goal is a responsive demo on a normal development machine.

---

# 26. RAG QUALITY

Implement sensible chunking.

Do not create one giant chunk per document.

Use overlapping chunks.

Store metadata with every chunk:

{
"category": "plane",
"source": "plane_additional_demo.json",
"chunk_id": "...",
"text": "..."
}

Retrieval should:

1. Filter by category.
2. Perform semantic similarity search.
3. Return top-k relevant chunks.
4. Pass those chunks to the LLM.

Keep top-k configurable.

---

# 27. IMPORTANT CATEGORY ISOLATION TEST

Create a simple test demonstrating:

A Plane question should retrieve Plane documents.

A Ship question should retrieve Ship documents.

A Bridge question should retrieve Bridge documents.

A Plane document should not accidentally become the primary source for a Ship query.

This is important because category-aware retrieval is one of the core features.

---

# 28. RAG DEMONSTRATION TEST

Create a simple test/demo procedure:

BASE DOCUMENT:

Plane:
"Demo information A."

QUESTION:

"What information is available about this object?"

ANSWER:

Uses "Demo information A."

Then upload:

Plane Additional Document:
"Demo information B."

Ask the same question again.

ANSWER should now include information from B.

Make sure this actually works before considering the application complete.

---

# 29. FACULTY-FRIENDLY UX

Assume the person using the application knows nothing about:

* YOLO
* embeddings
* vector databases
* RAG
* LLMs

Therefore the interface should explain the workflow visually.

For example:

Step 1
Upload an aerial image

↓

Step 2
The AI identifies the object

↓

Step 3
Select or confirm the detected category

↓

Step 4
Ask a question

↓

Step 5
The system retrieves relevant knowledge

↓

Step 6
The AI generates the answer

Keep technical terminology inside an optional "Technical Details" section.

---

# 30. SIDEBAR

Create a useful sidebar containing:

Application Status

Detection Model
✓ Loaded

Knowledge Base
✓ Loaded

Vector Store
✓ Ready

LLM
✓ Connected

Uploaded Documents
3

This makes the application look polished and helps during demonstrations.

---

# 31. TECHNICAL DETAILS SECTION

Provide an expandable section for technically interested users.

Show:

* model name
* detected class
* confidence
* number of detections
* retrieval category
* number of retrieved chunks
* source documents
* embedding model
* vector-store type

Do not show raw internal logs by default.

---

# 32. README

Create a complete README explaining:

1. Project purpose
2. Architecture
3. Installation
4. Environment variables
5. How to start Streamlit
6. How to upload an image
7. How to upload a document
8. How category-aware RAG works
9. How to reproduce the base-vs-uploaded-document demonstration
10. Troubleshooting

The primary command should ideally be:

streamlit run app.py

---

# 33. ACCEPTANCE CRITERIA

The project is complete only when all of the following work:

### Detection

* Streamlit starts successfully.
* Phase 1 model loads.
* User can upload an aerial image.
* Image is displayed.
* Detection runs.
* Bounding boxes/annotations are displayed.
* All detected objects are listed.
* Confidence scores are displayed.
* 15 DOTA categories are supported.

### RAG

* Base JSON knowledge loads automatically.
* Documents can be uploaded.
* User can select a category for an uploaded document.
* Documents are chunked.
* Embeddings are generated.
* Chunks are stored in a local vector store.
* Retrieval is category-aware.
* Questions can be asked.
* LLM generates an answer from retrieved context.
* Sources are displayed.

### Main demonstration

The following must work:

Image
→ Plane detected
→ Ask question
→ Answer from base knowledge

Then:

Upload new Plane document
→ Index document
→ Ask SAME question
→ Answer changes using the new information

### UI

* Clean
* intuitive
* faculty-friendly
* visually organized
* understandable without technical knowledge

### Engineering

* No unnecessary FastAPI.
* No unnecessary microservices.
* No hard-coded secrets.
* Heavy models are cached.
* Errors are handled gracefully.
* README is complete.

---

# FINAL IMPORTANT INSTRUCTION

Do not build additional features just because they are technically interesting.

This is an MVP.

The success criterion is NOT:

"Build the most sophisticated RAG system."

The success criterion is:

"Show a convincing end-to-end demonstration where an aerial image is classified into one of the 15 DOTA categories, that category controls retrieval, and uploading new category-specific documents changes the generated answer."

Prioritize making THAT workflow extremely reliable and polished.

Before finishing, run the application locally and test the complete workflow end-to-end.

Do not merely create files and assume they work.
