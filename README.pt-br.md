# Prof. Girafales 🧮

> Assistente de Professor de Matemática com IA — agente fine-tuned com RAG para diagnóstico implícito de dificuldades de alunos na plataforma FARMA.educacional.

🇺🇸 [English README here](README.md)

---

## Visão Geral

O **Prof. Girafales** é um agente de IA especializado em matemática, desenvolvido para operar como assistente pedagógico inteligente na plataforma [FARMA.educacional](https://farma.educacional.com.br). O sistema vai além de um chatbot simples: ele **infere as dificuldades implícitas** do aluno a partir do histórico de acertos e erros, sem depender de autoavaliação explícita.

**Exemplo:** se um aluno acerta a maioria dos exercícios de Cálculo mas consistentemente erra os que envolvem logaritmos, o agente identifica que o problema não é "Cálculo" em geral, mas especificamente o conceito de logaritmo — e adapta as respostas com esse contexto em mente.

---

## Motivação

Plataformas de ensino de matemática possuem dados ricos sobre o comportamento de alunos (acertos, erros, tempo de resposta, padrões de tentativa), mas raramente transformam esses dados em **contexto pedagógico acionável**. O Prof. Girafales preenche essa lacuna ao:

1. Modelar o perfil de dificuldades de cada aluno com base no histórico do FARMA.
2. Usar esse perfil como contexto aumentado na geração de respostas (RAG).
3. Aplicar um modelo fine-tuned para raciocínio matemático, com capacidade de seguir o pensamento passo a passo do aluno.

---

## Arquitetura

O projeto segue **Clean Architecture** com separação clara em camadas:

```
src/
├── domain/               # Núcleo do negócio (sem dependências externas)
│   ├── entities/         # Entidades: Aluno, Exercício, Diagnóstico
│   ├── value_objects/    # Objetos de valor: PontuacaoConceito, PadraoDificuldade
│   └── repositories/     # Interfaces (contratos) dos repositórios
│
├── application/          # Casos de uso e orquestração
│   ├── use_cases/        # Ex: DiagnosticarDificuldade, GerarResposta
│   └── services/         # Ex: PerfilAlunoService, RAGService
│
├── infrastructure/       # Implementações concretas
│   ├── database/
│   │   ├── models/       # Modelos ORM
│   │   └── repositories/ # Implementações dos repositórios
│   ├── ports/            # Adaptadores externos (FARMA API, LLM, etc.)
│   └── r_model/
│       ├── core/         # Inferência do modelo fine-tuned
│       └── training/     # Pipeline de fine-tuning
│
└── interface/
    └── http/
        └── routes/       # Endpoints FastAPI
```

---

## Stack Técnica

| Camada | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Validação | Pydantic v2 |
| Vector Store | Pinecone |
| Embeddings | sentence-transformers |
| Reranking | Cross-encoder (sentence-transformers) |
| LLM Base | A definir (OpenAI / Ollama / HuggingFace) |
| Fine-tuning | A definir (LoRA / QLoRA) |
| Cache | Redis |
| Processamento PDF | PyMuPDF + pypdf |
| Testes | pytest + pytest-asyncio |

---

## Conceitos-Chave

### RAG (Retrieval-Augmented Generation)
O sistema recupera contexto relevante de duas fontes antes de gerar uma resposta:
- **Base de conhecimento matemático** (apostilas, livros, exercícios resolvidos)
- **Perfil de dificuldades do aluno** (construído a partir do histórico no FARMA)

### Retrieval Argument Thought
Técnica de avaliação matemática que força o modelo a externalizar o raciocínio passo a passo antes de emitir uma resposta final, aumentando a precisão em problemas que exigem dedução formal.

### Fine-tuning para Compreensão Implícita
O modelo é fine-tuned especificamente para:
- Identificar o conceito matemático por trás de cada erro (não apenas "errou" vs "acertou")
- Adaptar o nível de explicação ao perfil inferido do aluno
- Manter coerência pedagógica ao longo de uma sessão

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/beatrizamante/farma-assistant.git
cd farma-assistant

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com suas credenciais
```

---

## Execução

```bash
uvicorn src.interface.http.app:app --reload --host 0.0.0.0 --port 8000
```

A documentação interativa da API ficará disponível em `http://localhost:8000/docs`.

---

## Testes

```bash
pytest src/__tests__/ -v --cov=src
```

---

## Status do Projeto

| Módulo | Status |
|---|---|
| Estrutura base (Clean Architecture) | ✅ |
| Definição de entidades do domínio | 🔧 Em andamento |
| Pipeline de RAG | 🔧 Em andamento |
| Fine-tuning do modelo | ⏳ Planejado |
| Integração FARMA.educacional | ⏳ Planejado |
| API REST completa | ⏳ Planejado |
| Testes | ⏳ Planejado |

---

## Contexto Acadêmico

Este projeto é desenvolvido como parte de uma pesquisa do curso de **Tecnologia em Sistemas para Internet (TSI)**, com foco na publicação de dois artigos:

1. **Revisão Literária** — Justificativa da escolha de arquitetura de IA, modelo base, técnicas de RAG e estratégia de fine-tuning para matemática.
2. **Artigo do Agente** — Descrição completa do Prof. Girafales: design, implementação, avaliação e resultados na plataforma FARMA.educacional.

---

## Licença

A definir.
