# STUDYARD — transcrição e resumo de aula

Data: 2026-08-26  
Status: aprovado em conversa; aguardando revisão do arquivo

## Objetivo

App local no Windows: gravar a aula do mestrado (microfone e/ou áudio do sistema), ir escrevendo uma **transcrição fiel** em Markdown durante a aula e, ao parar, gerar um **resumo** na mesma pasta. A UI é uma página em `localhost`. Transcrição e resumo usam uma API **compatível com OpenAI** configurada pelo usuário. Nada além dessa API sai da máquina.

## Fora de escopo (v1)

- Login, conta, sincronização em nuvem própria
- App mobile, `.exe` instalável (Tauri/Electron)
- Várias sessões ao mesmo tempo
- Diarização (separar professor e alunos)
- Captura 100% offline (modelos locais)
- Painel de configurações na UI (edita-se o JSON)

## Arquitetura

Um processo Python (FastAPI + Uvicorn) escuta só em `127.0.0.1` (porta padrão `8765`, configurável). Serve a UI estática e executa captura, sessão, cliente HTTP da API e escrita em disco.

O navegador **não** recebe a chave da API. Só envia comandos (gravar, parar, processar pendentes) e mostra status, último trecho e caminhos dos arquivos.

`config.json` na raiz do repositório (`D:\IA\STUDYARD\config.json`) é a única fonte de verdade para pasta, API e parâmetros. O servidor relê o JSON **ao clicar em Gravar** e ao processar pendentes, para não exigir restart após um ajuste.

## Componentes

| Peça | Faz | Depende de |
| --- | --- | --- |
| UI (`/`) | Gravar/Parar, fonte (mic / sistema / ambos), toggle salvar áudio (off por padrão), status, pasta do dia, processar pendentes | API HTTP local |
| Captura | PCM Windows WASAPI: mic, loopback do sistema, ou mistura dos dois em **um** stream mono 16 kHz | Dispositivos de áudio |
| Sessão | Cria pasta do dia, escolhe sufixo livre, append no `*_aula.md`, WAV temporário, resumo ao parar, pendências | Captura + cliente API + disco |
| Cliente API | `POST /v1/audio/transcriptions` e `POST /v1/chat/completions` no `api_base_url` do JSON | `config.json`, rede |

## Configuração

Arquivo: `D:\IA\STUDYARD\config.json` (não versionar com chave). Versionar `config.example.json` sem segredos.

Campos:

| Campo | Padrão | Significado |
| --- | --- | --- |
| `output_dir` | `D:\\IA\\STUDYARD\\transcricao` | Raiz das pastas por dia |
| `api_base_url` | `""` | Ex.: `https://api.openai.com/v1` (sem barra no fim) |
| `api_key` | `""` | Bearer da API |
| `transcription_model` | `whisper-1` | Modelo do endpoint de áudio |
| `summary_model` | `gpt-4o-mini` | Modelo do chat de resumo |
| `language` | `pt` | Idioma da transcrição |
| `chunk_seconds` | `25` | Tamanho da fatia ao vivo |
| `save_audio` | `false` | Se `true`, mantém o WAV final após sucesso |
| `port` | `8765` | Porta de `127.0.0.1` |

Gravar recusa começar se `api_base_url` ou `api_key` estiverem vazios, ou se `output_dir` não puder ser criado.

## Arquivos no disco

Pasta do dia (criada na primeira gravação daquele dia):

`{output_dir}\aaaa-mm-dd\`

Exemplos com `output_dir` padrão, em 2026-08-26:

- `D:\IA\STUDYARD\transcricao\2026-08-26\2026-08-26_aula.md`
- `D:\IA\STUDYARD\transcricao\2026-08-26\2026-08-26_resumo.md`

Segunda sessão no mesmo dia: `2026-08-26_aula-2.md` e `2026-08-26_resumo-2.md`; a terceira usa `-3`, e assim por diante. O número é o menor inteiro ≥ 1 cujo stem ainda não existe como `*_aula.md` ou `*_aula.wav` (`-1` não se escreve; a primeira sessão não tem sufixo). Uma sessão pendente (só aula, sem resumo) **ocupa** o stem.

WAV, quando existir, usa o mesmo stem: `2026-08-26_aula.wav` / `2026-08-26_aula-2.wav`.

Pendência de rede: `{stem}.pending.json` ao lado do WAV e do `*_aula.md` parcial.

O `*_aula.md` começa com um cabeçalho curto (data, fonte de áudio) e o corpo é texto corrido da transcrição, **acrescentado** a cada chunk. Não é necessário timestamp em cada linha na v1.

O `*_resumo.md` é gerado só no fim (ou no processar pendentes): tópicos, definições, exemplos e o que o professor enfatizou, em português, a partir da transcrição fiel — não é segundo transcript.

## Fluxo ao vivo

1. Gravar → uma sessão ativa (segundo Gravar é ignorado).
2. Cria pasta do dia e o próximo `*_aula.md`.
3. Abre captura; **sempre** grava WAV temporário da sessão (seguro), mesmo com `save_audio: false`.
4. A cada `chunk_seconds`: envia a fatia à transcrição; append do texto no `*_aula.md`; UI mostra status, último trecho e caminho da pasta.
5. Parar → flush do último chunk (pode ser mais curto).
   - Se **nenhum** chunk falhou: gera `*_resumo.md` via chat.
   - Se **algum** chunk falhou e a API responde: transcreve o WAV inteiro, substitui o corpo de `*_aula.md`, depois gera o resumo (mesmo critério de “processar pendentes”).
   - Se a API não responde: não apaga o WAV; grava `{stem}.pending.json`.
   - Sucesso: se `save_audio` da sessão é false, apaga o WAV; se true, deixa o WAV com o nome final.
6. Status final: “salvo em …” com os caminhos dos `.md`.

Uma sessão por vez. Fonte (mic / sistema / ambos) e o toggle de áudio da UI valem para aquela sessão; o toggle inicia igual a `save_audio` do JSON e pode ser mudado **antes** de Gravar.

## Perda de conexão e pendentes

Enquanto grava, o WAV temporário é a cópia fiel da aula, independente da API.

- Chunk de transcrição: até **2 retries** com espera curta; depois escreve `[trecho ~mm:ss não transcrito]` no `.md`, avisa na UI e **continua gravando**.
- Se a API/rede estiver indisponível: UI “offline / API indisponível”; áudio segue no WAV.
- Ao Parar sem transcrição completa ou sem resumo: **não apaga o WAV**. Grava `{stem}.pending.json` com `need` (`transcribe`, `summarize`, ou ambos) e `save_audio` daquela sessão.
- Sessão bem-sucedida: remove `.pending.json` se houver; WAV só permanece se `save_audio` da sessão estiver on.

**Processar pendentes** (quando a rede voltar): para cada `{stem}.pending.json`, o WAV é a fonte da verdade.

- Se falta transcrever: transcreve o **WAV inteiro** e **substitui** o corpo de `*_aula.md` (cabeçalho preservado). Isso evita buracos e duplicar o que já tinha sido append ao vivo.
- Em seguida gera `*_resumo.md`.
- Sucesso: apaga `.pending.json`; apaga o WAV se `save_audio` **da pendência** for false.

Se o resumo falhar mas a transcrição estiver ok: `*_aula.md` permanece; pendência só de `summarize`; a UI oferece gerar resumo de novo sem regravar.

## Outros erros

- `config.json` inválido ou pasta inacessível: Gravar não começa; mensagem na página pedindo ajuste no JSON.
- Dispositivo de áudio some: para a captura, flush, avisa; não apaga `.md` nem o WAV; vira pendência se a transcrição/resumo não fechou.
- Crash do processo: o que já foi append no `*_aula.md` e o WAV (se o OS não tiver truncado) ficam no disco. Na próxima abertura, a UI lista: (1) todo `{stem}.pending.json`; (2) todo `*_aula.wav` sem `*_resumo.md` correspondente — nesse caso cria `.pending.json` se ainda não existir (`transcribe` se o `*_aula.md` estiver vazio ou tiver marcadores de falha; senão só `summarize`). Não dispara resumo sozinho.

## Contrato da API

Transcrição: `POST {api_base_url}/v1/audio/transcriptions` (multipart, modelo e `language` do JSON).

Resumo: `POST {api_base_url}/v1/chat/completions` com a transcrição completa e um system prompt fixo: produzir Markdown de estudo em português (estrutura, definições, ênfases), sem inventar conteúdo que não esteja na transcrição. Ignorar linhas `[trecho … não transcrito]` se ainda existirem.

Trocar de provedor = mudar `api_base_url`, `api_key` e nomes dos modelos no JSON.

## UI

Página única em `http://127.0.0.1:{port}`:

- Fonte de áudio: mic / sistema / ambos
- Toggle salvar áudio (padrão off)
- Gravar / Parar
- Status: ocioso, gravando, transcrevendo chunk, gerando resumo, offline, erro
- Caminho da pasta do dia e nomes dos arquivos da sessão
- Lista de pendentes + botão processar

Sem editor de chave na página.

## Testes

**Automáticos** (captura e API mockadas):

- Primeira sessão do dia e sufixos `-2`, `-3`; criação de `transcricao\aaaa-mm-dd`
- Append de chunks na ordem; não reescreve o arquivo no caminho ao vivo
- Recusa Gravar sem `api_base_url`/chave ou com pasta inválida
- Três falhas de chunk → marcador no `.md` e sessão segue
- Sucesso + `save_audio` false → WAV temporário removido; true → WAV final na pasta
- Rede caiu → WAV + `.pending.json`; processar pendentes completa aula + resumo e limpa pendência
- Resumo falhou → `aula.md` intacto; retry só o chat

**Manuais no Windows:**

- Mic; áudio do sistema (Zoom ou YouTube); ambos
- UI: status, pasta, Gravar → Parar → arquivos no Explorer
- Wi‑Fi off no meio → Parar → religar → processar pendentes

## Critério de pronto

Em uma aula real (ou simulada), com `config.json` apontando para uma API compatível: a pasta do dia existe, o `*_aula.md` cresce durante a gravação, o `*_resumo.md` aparece ao parar, a UI mostra o caminho, e uma queda de rede no meio permite recuperar transcrição e resumo depois a partir do WAV.
