# STUDYARD

App local no Windows: grava a aula (microfone e/ou áudio do sistema), escreve a transcrição em Markdown **durante** a aula e gera o resumo ao parar.

## Requisitos

- Windows 10+
- Python 3.11+
- Chave de uma API compatível com OpenAI **só para o resumo** (a transcrição é local com faster-whisper, modelo `base`, português)

## Configuração

1. Copie `config.example.json` para `config.json` na raiz deste repositório.
2. Preencha `api_key` e, se não for OpenAI, `api_base_url` e `summary_model`. A transcrição usa `whisper_model` (padrão `base`) via faster-whisper na CPU.
3. `output_dir` padrão: `D:\IA\STUDYARD\transcricao` (pastas `aaaa-mm-dd` por dia de aula).

Não versione `config.json` (já está no `.gitignore`).

## Como rodar

```powershell
cd D:\IA\STUDYARD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m studyard
```

Abra `http://127.0.0.1:8765` (porta em `config.json`).

Arquivos gerados:

- `transcricao\aaaa-mm-dd\aaaa-mm-dd_aula.md` — transcrição (cresce ao vivo)
- `transcricao\aaaa-mm-dd\aaaa-mm-dd_resumo.md` — resumo ao parar
- Segunda aula no mesmo dia: `..._aula-2.md` / `..._resumo-2.md`

Áudio WAV só permanece se você marcar “Salvar áudio”. Se o resumo (API) falhar, o WAV é guardado para **Processar pendentes**. A transcrição ao vivo não depende da rede.

## Testes

```powershell
python -m pytest tests -v
```

## Checklist manual

- Microfone só; áudio do sistema (Zoom ou YouTube); ambos
- Status e caminho da pasta na página; Gravar → Parar → arquivos no Explorer
- Wi‑Fi off no meio → Parar → religar → Processar pendentes
