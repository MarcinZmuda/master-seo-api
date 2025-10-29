{
  "openapi": "3.1.1",
  "info": {
    "title": "Unified SEO API v5.3 (Hybrid)",
    "version": "5.3.0",
    "description": "API do workflow SEO (S1–S4) oparte na Firestore. Endpoint /createProject jest hybrydowy: akceptuje 'text/plain' (preferowane) lub 'application/json' (fallback)."
  },
  "servers": [
    {
      "url": "https://master-seo-api.onrender.com",
      "description": "Production server for Unified SEO API v5.3"
    }
  ],
  "paths": {
    "/api/s1_analysis": {
      "post": {
        "summary": "SEO – Analiza SERP, treści i n-gramów (S1)",
        "operationId": "performS1Analysis",
        "description": "Narzędzie do analizy konkurencji. Pobiera top 5 wyników Google, analizuje treść i n-gramy.",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "topic": {
                    "type": "string",
                    "description": "Fraza kluczowa lub temat SEO do analizy"
                  }
                },
                "required": [
                  "topic"
                ]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Zwraca raport analizy."
          }
        }
      }
    },
    "/api/project/create": {
      "post": {
        "summary": "Tworzy nowy projekt SEO w bazie danych (S2) [HYBRYDOWY]",
        "operationId": "createProject",
        "description": "Przetwarza brief (jako surowy tekst LUB JSON), parsuje go i zapisuje słowa kluczowe w bazie danych. Zwraca ID projektu.",
        "requestBody": {
          "required": true,
          "content": {
            "text/plain": {
              "schema": {
                "type": "string",
                "description": "Surowy tekst briefu (H2/BASIC/EXTENDED). To jest preferowana metoda."
              }
            },
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "h2_terms": {
                    "type": "array",
                    "items": { "type": "string" }
                  },
                  "basic_terms": {
                    "type": "object",
                    "additionalProperties": { "type": "string" }
                  },
                  "extended_terms": {
                    "type": "object",
                    "additionalProperties": { "type": "string" }
                  }
                },
                "description": "Opcjonalna metoda wysłania briefu jako JSON."
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Projekt utworzony pomyślnie.",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "project_id": { "type": "string" },
                    "keywords_parsed": { "type": "integer" }
                  }
                }
              }
            }
          },
          "400": {
            "description": "Błąd parsowania briefu lub brak danych."
          }
        }
      }
    },
    "/api/project/{project_id}/add_batch": {
      "post": {
        "summary": "Dodaje batch tekstu i przelicza stan słów kluczowych (S3)",
        "operationId": "addBatchToProject",
        "description": "Dodaje tekst batcha (jako text/plain) do projektu i zwraca zaktualizowany raport stanu.",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": { "type": "string" }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "text/plain": {
              "schema": {
                "type": "string",
                "description": "Surowy tekst nowo napisanego batcha."
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Zwraca raport tekstowy (lista stringów).",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": { "type": "string" }
                }
              }
            }
          }
        }
      }
    },
    "/api/project/{project_id}": {
      "delete": {
        "summary": "Usuwa projekt z bazy danych po zakończeniu (S4)",
        "operationId": "deleteProject",
        "description": "Trwale usuwa dokument projektu z bazy Firestore.",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": { "type": "string" }
          }
        ],
        "responses": {
          "200": {
            "description": "Projekt został pomyślnie usunięty."
          }
        }
      }
    }
  }
}
