# LangChain RAG Chatbot with OpenAI + ChromaDB

Este proyecto es una implementación práctica de un sistema **RAG** (Retrieval-Augmented Generation) utilizando **LangChain**, **ChromaDB** y **OpenAI**. Permite consultar documentos locales (en este caso, un `.md` del libro *Alice in Wonderland*) usando lenguaje natural, y obtener respuestas generadas por una IA con base en el contenido real.

---

## 🧠 ¿Qué es RAG?

RAG combina:

- 🔍 Recuperación de información (vector search): busca los fragmentos más relevantes en tus documentos.
- 🧠 Generación con LLM (modelo de lenguaje): genera una respuesta en lenguaje natural basándose en esos fragmentos.

---

## 🛠️ Requisitos previos

- Python 3.10+  
- Cuenta en [OpenAI](https://platform.openai.com/) y una API key válida.
- Editor de texto (VS Code recomendado).
- Conexión a Internet para las llamadas a OpenAI.

---

## 📦 Instalación paso a paso

1. **Clona este repositorio y entra en el proyecto**

```bash
git clone https://github.com/tu_usuario/tu_repo.git
cd rag-project
```

2. **Crea y activa un entorno virtual**

```bash
python -m venv venv
.env\Scriptsctivate   # En Windows
source venv/bin/activate # En macOS/Linux
```

3. **Instala las dependencias principales**

> ⚠️ Si usas Windows y obtienes errores con `onnxruntime`, revisa más abajo los pasos específicos.

```bash
pip install -r requirements.txt
```

4. **Instala soporte para Markdown (muy importante para leer los `.md`)**

```bash
pip install "unstructured[md]"
```

5. **Crea un archivo `.env` y añade tu clave de OpenAI:**

```
OPENAI_API_KEY=sk-xxxxxx
```

---

## 🪛 Posibles errores y soluciones

### ❌ `Permission denied` (en Windows)
➡ Ejecuta `pip install` con `--no-cache-dir`:
```bash
pip install chromadb --no-cache-dir
```

### ❌ `ModuleNotFoundError: No module named 'unstructured'`
➡ Instala con:
```bash
pip install "unstructured[md]"
```

### ❌ `LangChainDeprecationWarning: predict is deprecated`
➡ Usa `.invoke()` en lugar de `.predict()` en tus scripts.

### ❌ `libmagic is unavailable`
➡ Es solo una advertencia. Puedes ignorarla, pero si quieres soporte completo en detección de archivos, instala `libmagic` manualmente (opcional).

---

## 🧪 Uso del sistema

### 1. Crear la base de datos vectorial (embedding + almacenamiento)

```bash
python create_database.py
```

Esto divide el documento en chunks y los guarda como vectores en `chroma/`.

### 2. Consultar a la IA con tu propia pregunta

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

---

## 🌐 Integración con Frontend en React

Si ya tienes un frontend con React (por ejemplo, un chat que use OpenAI), puedes conectar este backend RAG usando **FastAPI**.

### Backend con FastAPI

Instala FastAPI y Uvicorn:

```bash
pip install fastapi uvicorn
```

Crea un archivo `backend.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

class Question(BaseModel):
    query: str

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = '''
Answer the question using only the following context:

{context}

---

Question: {question}
'''

@app.post("/ask")
async def ask_question(data: Question):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(data.query, k=3)

    if not results or results[0][1] < 0.7:
        return {"answer": "No relevant information found."}

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=data.query)
    model = ChatOpenAI()
    response = model.invoke(prompt).content
    return {"answer": response}
```

Lanza el backend:

```bash
uvicorn backend:app --reload
```

### En tu frontend React:

Haz una llamada al backend:

```js
const response = await fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Your user question" })
});

const data = await response.json();
console.log(data.answer);
```

---

## 📁 Estructura esperada del proyecto

```
rag-project/
├── chroma/                    ← base de datos vectorial
├── data/
│   └── books/
│       └── alice_in_wonderland.md
├── .env
├── create_database.py
├── query_data.py
├── compare_embeddings.py
├── backend.py
├── requirements.txt
└── README.md
```

---

## 📹 Tutorial original en video

Basado en este excelente tutorial:  
🎥 [RAG + Langchain Python Project - YouTube](https://www.youtube.com/watch?v=tcqEUSNCn8I)

---

## ✅ Créditos y mejoras

Proyecto extendido por Oscar con mejoras de compatibilidad, corrección de deprecaciones, conexión con frontend y documentación clara.  
¡Listo para usar en tus propios documentos, manuales, libros o FAQs!
