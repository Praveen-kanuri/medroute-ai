roject name

MedRoute AI
An Open-Model Multimodal Healthcare Appointment Assistant

GitHub repository:

medroute-ai

The system assists with specialty routing and appointment booking. It should never claim to diagnose or prescribe treatment.

Initial model strategy

One technical clarification: Groq is the inference platform hosting several open-weight models. Because Deepgram is being accessed through hosted APIs, the complete application is better described as open-model-first, rather than fully open-source.

Component	Initial choice	Experiment
Text LLM	openai/gpt-oss-20b on Groq	Compare with llama-3.3-70b-versatile and openai/gpt-oss-120b
Structured symptom extraction	openai/gpt-oss-20b	Groq currently supports strict JSON-schema output for GPT-OSS 20B and 120B
Uploaded/push-to-talk STT	whisper-large-v3-turbo on Groq	Compare accuracy with whisper-large-v3
Real-time streaming STT	Deepgram flux-general-en	Add after push-to-talk works
Text-to-speech	Deepgram Aura-2	Start with aura-2-thalia-en
Orchestration	LangGraph	Safety, routing, doctor search and booking nodes

Groq recommends whisper-large-v3-turbo for price/performance and whisper-large-v3 when accuracy is more important. For our POC, Turbo should be the default, and we'll measure both against medical phrases, doctor names and specialty terminology. Groq speech-to-text documentation

For live conversation, Deepgram Flux is more appropriate than the Groq Whisper file-transcription endpoint because Flux supports streaming and conversational end-of-turn detection. Deepgram Flux documentation

Voice experiments
Push-to-talk:
Audio → Groq Whisper → Groq LLM → Deepgram Aura-2

Real-time conversation:
Microphone → Deepgram Flux → Groq LLM → Deepgram Aura-2
Development milestones
Repository and engineering foundation
Mock hospital, doctor and appointment APIs
Text-based symptom intake and specialty routing
LangGraph workflow with safety guardrails
Doctor availability and simulated booking
Push-to-talk voice interaction
Real-time streaming voice
Image-supported routing
Evaluation, Docker deployment and portfolio documentation
Initial repository structure
medroute-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── graph/
│   │   ├── providers/
│   │   │   ├── llm/
│   │   │   ├── speech_to_text/
│   │   │   └── text_to_speech/
│   │   ├── safety/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── tools/
│   │   └── main.py
│   └── tests/
├── frontend/
├── data/
│   └── synthetic/
├── docs/
│   ├── architecture.md
│   ├── model-experiments.md
│   ├── safety-design.md
│   ├── prompts-used.md
│   └── roadmap.md
├── .claude/
│   └── skills/
├── .github/
│   └── workflows/
├── CLAUDE.md
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
└── README.md

Recommended phases
Phase	User input	Chatbot capability	Output
1 — Text MVP	Symptoms in text	Extract symptoms, check urgency, identify specialty, query available doctors and appointment slots	Text response
2 — Multimodal	Text plus supported images	Analyze the information only for routing, combine it with symptom details, and identify the appropriate specialty	Text and optional audio
3 — Voice Assistant	Spoken request	Speech-to-text, conversational routing, doctor search, slot selection, and confirmation	Spoken response
4 — Complete Booking	Text, image, or voice	Verify patient preferences, insurance, location, and availability; reserve the selected slot	Booking confirmation

Suggested architecture
Frontend: React or Streamlit
Backend: FastAPI
Orchestration: LangGraph
LLM: Claude through AWS Bedrock or an OpenAI model
Doctor and appointment data: PostgreSQL
Knowledge base: Specialty descriptions, hospital policies, preparation instructions and FAQs
Retrieval: Bedrock Knowledge Bases, Pinecone, or pgvector
Tools: Doctor search, availability lookup, appointment booking and cancellation
Voice: Speech-to-text and text-to-speech services
Observability: LangSmith or CloudWatch
Deployment: Docker with AWS ECS Fargate

A good LangGraph design would contain nodes such as:

Safety check → Symptom extraction → Specialty routing → Doctor search → Availability lookup → Recommendation → Booking confirmation

The LLM should return structured data like:

{
  "symptoms": ["knee pain", "swelling"],
  "duration": "five days",
  "urgency": "routine",
  "recommended_specialty": "Orthopedics",
  "location": "Dallas",
  "preferred_time": "weekday morning"
}

Then deterministic tools—not the LLM—should query the doctor database and confirm availability. This prevents hallucinated doctors and appointment slots.

For the portfolio, start only with Phase 1 and make it production-like. Include mock doctor schedules, specialty routing, real tool calls, booking confirmation, safety guardrails, evaluation metrics, and a polished interface. That will demonstrate more engineering ability than attempting text, image, video, and voice simultaneously.
