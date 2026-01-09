# AstroAPI Architecture & Integration Guide

## 1. Visão geral
AstroAPI é uma API FastAPI para cálculos astrológicos (mapa natal, trânsitos, clima cósmico) com suporte a Swiss Ephemeris (`swisseph`) e recursos de IA via OpenAI. O serviço expõe endpoints REST e aplica autenticação via API key e `X-User-Id` para todos os endpoints protegidos.

**Principais capacidades:**
- Cálculo de mapa natal com casas, cúspides e planetas.
- Cálculo de trânsitos e aspectos entre planetas do mapa natal e do trânsito.
- Clima cósmico diário e intervalo (fase e signo lunar).
- Renderização de dados de mapa para uso em front-ends.
- Diagnóstico de efemérides (comparação direta com Swiss Ephemeris).
- Chat com IA usando payload astrológico.

**Tecnologias principais:**
- FastAPI, Pydantic, OpenAI SDK.
- Swiss Ephemeris (`swisseph`).
- Cache interno (`core.cache`).
- Auth via API key (`core.security`).

## 2. Mapa de módulos
- **`main.py`**: app FastAPI, modelos Pydantic, endpoints, middleware, logs, validações e orquestração de fluxos.
- **`astro/ephemeris.py`**: cálculos centrais com Swiss Ephemeris (`compute_chart`, `compute_transits`, `compute_moon_only`).
- **`astro/aspects.py`**: cálculo de aspectos entre planetas.
- **`astro/utils.py`**: utilitários de astronomia (e.g. `to_julian_day`, `deg_to_sign`, `angle_diff`).
- **`core/security.py`**: validação de API key e user id.
- **`core/cache.py`**: cache com TTLs.
- **`core/plans.py`**: regras de plano (ex.: trial/premium).
- **`ai/prompts.py`**: construção de prompts para o chat com IA.
- **`tests/`**: testes de integração e de validação da efeméride.

## 3. Fluxos de alto nível
### 3.1 Cálculo de mapa natal
1. `POST /v1/chart/natal` recebe dados de nascimento e localização.
2. `_tz_offset_for` resolve timezone IANA ou usa `tz_offset_minutes`.
3. `compute_chart` calcula casas, cúspides e planetas via Swiss Ephemeris.
4. Resposta cacheada por `TTL_NATAL_SECONDS`.

### 3.2 Trânsitos
1. `POST /v1/chart/transits` recebe dados de nascimento e `target_date`.
2. Resolve timezone e calcula mapa natal com `compute_chart`.
3. `compute_transits` calcula planetas para o dia alvo.
4. `compute_transit_aspects` compara planetas natal x trânsito.
5. Enriquecimento com clima cósmico do dia (fase e signo lunar).

### 3.3 Clima cósmico
- `GET /v1/cosmic-weather` e `GET /v1/cosmic-weather/range` usam `compute_moon_only`.
- Cache por usuário + data.

### 3.4 Renderização de dados do mapa
- `POST /v1/chart/render-data` gera dados simplificados para uso visual no front-end (planetas, casas, zodíaco).

### 3.5 Diagnóstico de efemérides
- `POST /v1/diagnostics/ephemeris-check` compara posições calculadas pelo `compute_chart` com resultados diretos do `swisseph.calc_ut`.

### 3.6 Chat com IA
- `POST /v1/ai/cosmic-chat` usa `OPENAI_API_KEY` e `ai/prompts.py`.
- Envia payload astrológico e retorna resposta textual.

## 4. Endpoints (resumo)
### Públicos
- `GET /` → status básico do serviço.
- `GET /health` → ok.

### Auxiliares
- `POST /v1/time/resolve-tz` → resolve offset a partir de timezone IANA.
- `POST /v1/diagnostics/ephemeris-check` → validação das posições.

### Astrologia
- `POST /v1/chart/natal`
- `POST /v1/chart/transits`
- `POST /v1/chart/render-data`
- `GET /v1/cosmic-weather`
- `GET /v1/cosmic-weather/range`

### Conteúdo/alertas
- `GET /v1/alerts/system`
- `GET /v1/notifications/daily`

### IA
- `POST /v1/ai/cosmic-chat`

## 5. Integração com Front-end
### 5.1 Autenticação
A maioria dos endpoints exige:
- Header `Authorization: Bearer <API_KEY>`
- Header `X-User-Id: <user_id>`

Exemplo (fetch):
```ts
await fetch(`${API_URL}/v1/chart/natal`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${API_KEY}`,
    "X-User-Id": userId,
  },
  body: JSON.stringify(payload),
});
```

### 5.2 Fluxo típico no front-end
1. **Resolver timezone** (opcional): `POST /v1/time/resolve-tz`.
2. **Mapa natal**: `POST /v1/chart/natal`.
3. **Renderização visual**: `POST /v1/chart/render-data` (gera casas/planetas em formato fácil para UI).
4. **Trânsitos**: `POST /v1/chart/transits` para períodos alvo.
5. **Clima cósmico**: `GET /v1/cosmic-weather`.
6. **IA**: `POST /v1/ai/cosmic-chat` usando o payload calculado.

### 5.3 Boas práticas no front-end
- Cachear respostas do backend quando possível (principalmente mapas). 
- Usar `house_system`, `zodiac_type` e `ayanamsa` conforme configurações do usuário.
- Tratar erros 400 (timezone inválido) e 500 (cálculo falhou) exibindo mensagens amigáveis.
- Para UX, considerar loading states para chamadas com cálculo astrológico.

## 6. Modelos e enums relevantes
- `HouseSystem` (main.py) controla códigos de sistema de casas.
- `ZodiacType` define `tropical` ou `sidereal`.
- `NatalChartRequest`, `TransitsRequest`, `RenderDataRequest` definem payloads principais.

## 7. Configuração e variáveis de ambiente
- `OPENAI_API_KEY`: obrigatório para `/v1/ai/cosmic-chat`.
- `API_KEY`: usado na autenticação (via `core.security`).
- `ALLOWED_ORIGINS`: CORS.
- `LOG_LEVEL`: nível de logs.
- `OPENAI_MODEL`, `OPENAI_MAX_TOKENS_FREE/PAID`: configuração do chat.

## 8. Observabilidade e logs
- Middleware registra `request_id`, path, status, latência.
- Logs estruturados em JSON.

## 9. Cache e performance
- `TTL_NATAL_SECONDS`, `TTL_TRANSITS_SECONDS`, `TTL_RENDER_SECONDS`, `TTL_COSMIC_WEATHER_SECONDS`.
- Cache é aplicado por usuário + payload para endpoints críticos.

## 10. Riscos e considerações
- Timezone inválido retorna 400.
- Erros de Swiss Ephemeris podem impactar cálculos.
- Dependência do OpenAI em `/v1/ai/cosmic-chat`.

## 11. Checklist de integração
- [ ] Definir `API_URL` no front-end.
- [ ] Configurar headers de auth.
- [ ] Usar payloads válidos (timezone ou tz_offset).
- [ ] Tratar códigos de erro.
- [ ] Mapear respostas em componentes visuais.

