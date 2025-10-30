// master-seo-api.js
// v5.7 hybrid - Firestore + SerpApi (Google, AI Overview, AI Mode, Autocomplete)

import express from "express";
import fetch from "node-fetch";
import admin from "firebase-admin";
import nlp from "compromise";

const app = express();
app.use(express.json({ limit: "10mb" }));

// 🔑 SerpApi key
const SERP_API_KEY = "25c097e061a90518eb61b13510e54d8339d0459ad3c2d9d055d12a30df494f59";

// 🔥 Firestore setup
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.applicationDefault(),
  });
}
const db = admin.firestore();

// ===========================================================
// 🧠 S1_ANALYSIS — analiza konkurencji + AI Overview + AI Mode
// ===========================================================
app.post("/api/s1_analysis", async (req, res) => {
  try {
    const { topic } = req.body;
    if (!topic) return res.status(400).json({ error: "Missing topic" });

    // 1️⃣ Klasyczne wyniki Google SERP (top 3)
    const serp = await fetch(
      `https://serpapi.com/search.json?engine=google&q=${encodeURIComponent(topic)}&num=3&gl=pl&hl=pl&api_key=${SERP_API_KEY}`
    ).then(r => r.json());

    // 2️⃣ AI Overview (jeśli dostępne)
    let aiOverview = null;
    if (serp.ai_overview?.page_token) {
      aiOverview = await fetch(
        `https://serpapi.com/search.json?engine=google_ai_overview&page_token=${serp.ai_overview.page_token}&api_key=${SERP_API_KEY}`
      ).then(r => r.json());
    }

    // 3️⃣ AI Mode — alternatywny kontekst AI Overview
    const aiMode = await fetch(
      `https://serpapi.com/search.json?engine=google_ai_mode&q=${encodeURIComponent(topic)}&hl=pl&gl=pl&api_key=${SERP_API_KEY}`
    ).then(r => r.json());

    // 4️⃣ Google Autocomplete — podpowiedzi
    const autocomplete = await fetch(
      `https://serpapi.com/search.json?engine=google_autocomplete&q=${encodeURIComponent(topic)}&hl=pl&gl=pl&client=chrome&api_key=${SERP_API_KEY}`
    ).then(r => r.json());

    // =======================================================
    // 🧠 Przetwarzanie treści z SERP i AI Overview
    // =======================================================
    const textBlocks = [
      ...(serp.organic_results?.map(r => r.snippet) || []),
      ...(aiOverview?.ai_overview?.text_blocks?.map(b => b.snippet) || []),
      ...(aiMode?.results?.map(r => r.snippet || r.content || "") || []),
    ].join(" ");

    const doc = nlp(textBlocks);
    const ngrams = doc.ngrams({ size: [2, 3, 4] }).slice(0, 50);
    const entities = [...new Set(doc.topics().out("array"))];

    // =======================================================
    // 🧩 Kandydaci H2 (SERP + AI Overview + AI Mode)
    // =======================================================
    const h2Candidates = [
      ...(aiOverview?.ai_overview?.text_blocks
        ?.filter(b => b.type === "heading")
        ?.map(b => b.snippet) || []),
      ...(serp.organic_results?.flatMap(r =>
        (r.snippet || "")
          .split(". ")
          .slice(0, 1)
          .map(s => s.trim())
      ) || []),
      ...(aiMode?.results?.flatMap(r =>
        (r.heading || "")
          .split(/[\n.]/)
          .map(s => s.trim())
          .filter(Boolean)
      ) || []),
    ];

    // =======================================================
    // 🧭 Raport z analizy
    // =======================================================
    res.json({
      topic,
      identified_urls: (serp.organic_results || []).slice(0, 5).map(r => r.link),
      ai_overview_status: !!aiOverview,
      ai_mode_results: aiMode?.results?.length || 0,
      autocomplete_suggestions: autocomplete.suggestions?.map(s => s.value) || [],
      h2_suggestions: [...new Set(h2Candidates)].slice(0, 10),
      ngrams: ngrams.map(n => n.normal),
      entities,
    });
  } catch (err) {
    console.error("❌ S1 Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ===========================================================
// 🧩 S2 - CREATE PROJECT
// ===========================================================
app.post("/api/project/create", async (req, res) => {
  try {
    const { brief_base64 } = req.body;
    if (!brief_base64) return res.status(400).json({ error: "Missing brief_base64" });

    const decoded = Buffer.from(brief_base64, "base64").toString("utf-8");
    const keywords = decoded.match(/^(.+?):/gm)?.map(k => k.replace(":", "").trim()) || [];

    const headersCount = (decoded.match(/^={8,}/gm) || []).length;
    const project = await db.collection("seo_projects").add({
      created_at: new Date(),
      brief: decoded,
      keywords_state: keywords.map(k => ({ term: k, count: 0, status: "UNDER" })),
      headers_count: headersCount,
    });

    res.json({
      project_id: project.id,
      keywords_parsed: keywords.length,
      headers_count: headersCount,
    });
  } catch (err) {
    console.error("❌ Create Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ===========================================================
// ✍️ S3 - ADD BATCH
// ===========================================================
app.post("/api/project/:id/add_batch", async (req, res) => {
  try {
    const { id } = req.params;
    const text = req.body;
    if (!text) return res.status(400).json({ error: "Missing text body" });

    const projectRef = db.collection("seo_projects").doc(id);
    const project = await projectRef.get();
    if (!project.exists) return res.status(404).json({ error: "Project not found" });

    const { keywords_state } = project.data();

    // 🔢 Zlicz frazy
    let updated = keywords_state.map(k => {
      const regex = new RegExp(`\\b${k.term}\\b`, "gi");
      const matches = text.match(regex);
      const count = matches ? matches.length : 0;
      const total = (k.count || 0) + count;

      let status = "UNDER";
      if (total >= 3) status = "OVER";
      if (total >= 6) status = "LOCKED";

      return { ...k, count: total, status };
    });

    await projectRef.update({
      keywords_state: updated,
      last_batch: text,
      updated_at: new Date(),
    });

    res.json({
      message: "Batch saved and analyzed",
      keywords_report: updated,
      locked_terms: updated.filter(k => k.status === "LOCKED").map(k => k.term),
    });
  } catch (err) {
    console.error("❌ Add Batch Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ===========================================================
// 🧹 S4 - DELETE PROJECT
// ===========================================================
app.delete("/api/project/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await db.collection("seo_projects").doc(id).delete();
    res.json({ message: `✅ Projekt ${id} został pomyślnie usunięty.` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ===========================================================
// ❤️ HEALTH CHECK
// ===========================================================
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    version: "v5.7-hybrid-ai-mode",
    message: "Master SEO API działa poprawnie (AI Overview + AI Mode + Autocomplete aktywne).",
  });
});

// ===========================================================
// 🔛 START SERVER
// ===========================================================
app.listen(8080, () => {
  console.log("✅ Master SEO API (v5.7 hybrid) running on port 8080");
});
