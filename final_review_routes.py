"""
===============================================================================
🔍 FINAL REVIEW ROUTES v23.0 - Zintegrowany z UnifiedValidator
===============================================================================
Zmiany:
1. Używa UnifiedValidator zamiast własnych walidacji
2. Integracja z VersionManager dla rollbacków
3. Uproszczona logika
===============================================================================
"""

import os
import re
import traceback
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import google.generativeai as genai

from unified_validator import validate_content, full_validate, ValidationConfig, validate_eeat, validate_polish_quality
from version_manager import VersionManager, VersionSource

final_review_routes = Blueprint("final_review_routes", __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# 🔍 FINAL REVIEW v23.0
# ================================================================
@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    """
    Pełna walidacja końcowa artykułu.
    Używa UnifiedValidator dla spójności z preview/approve.
    """
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        batches = data.get("batches", [])
        keywords_state = data.get("keywords_state", {})
        main_keyword = data.get("main_keyword", data.get("topic", ""))
        s1_data = data.get("s1_data", {})
        
        if not batches:
            return jsonify({"error": "No batches to review"}), 400
        
        # Połącz wszystkie batche
        full_text = "\n\n".join(b.get("text", "") for b in batches if b.get("text"))
        word_count = len(full_text.split())
        
        # Pobierz ngrams z S1
        ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.5][:15]
        
        # ================================================================
        # WALIDACJA UŻYWAJĄC UnifiedValidator
        # ================================================================
        validation_result = validate_content(
            text=full_text,
            keywords_state=keywords_state,
            main_keyword=main_keyword,
            required_ngrams=ngrams,
            is_intro_batch=False,
            existing_lists_count=0,
            validation_mode="final"
        )
        
        # ================================================================
        # DODATKOWE SPRAWDZENIA SPECYFICZNE DLA FINAL REVIEW
        # ================================================================
        additional_issues = []
        recommendations = []
        
        # 1. Sprawdź brakujące keywords (które nie osiągnęły minimum)
        for rid, meta in keywords_state.items():
            actual = meta.get("actual_uses", 0)
            target_min = meta.get("target_min", 1)
            keyword = meta.get("keyword", "")
            kw_type = meta.get("type", "BASIC")
            
            if actual < target_min:
                missing = target_min - actual
                severity = "ERROR" if kw_type in ["MAIN", "BASIC"] else "WARNING"
                additional_issues.append({
                    "code": "MISSING_KEYWORD",
                    "severity": severity,
                    "keyword": keyword,
                    "actual": actual,
                    "target_min": target_min,
                    "missing": missing
                })
                recommendations.append(f"DODAJ '{keyword}' min. {missing}x")
        
        # 2. Sprawdź main vs synonyms (z validation_result)
        kw_analysis = validation_result.keywords_analysis
        if kw_analysis.get("main_ratio", 1.0) < 0.5:
            recommendations.append(
                f"Zamień synonimy na '{main_keyword}' - obecnie tylko {kw_analysis['main_ratio']:.0%}"
            )
        
        # 3. Sprawdź krótkie H3
        for section in validation_result.structure_analysis.get("h3_sections", []):
            if section["word_count"] < ValidationConfig.H3_MIN_WORDS:
                recommendations.append(
                    f"Rozbuduj H3 '{section['title']}' o {ValidationConfig.H3_MIN_WORDS - section['word_count']} słów"
                )
        
        # 4. Sprawdź brakujące n-gramy
        ngram_info = validation_result.structure_analysis.get("ngram_coverage", {})
        if ngram_info.get("missing"):
            recommendations.append(
                f"Wpleć n-gramy: {', '.join(ngram_info['missing'][:3])}"
            )
        
        # ================================================================
        # 5. E-E-A-T ANALYSIS (Google 2024+)
        # ================================================================
        eeat_analysis = validate_eeat(full_text)
        
        # Dodaj rekomendacje E-E-A-T jeśli wynik słaby
        if eeat_analysis.get("status") == "NEEDS_IMPROVEMENT":
            eeat_recs = eeat_analysis.get("recommendations", [])
            for rec in eeat_recs[:3]:
                if rec != "E-E-A-T OK":
                    recommendations.append(f"E-E-A-T: {rec}")
        
        # Dodaj issue jeśli E-E-A-T bardzo słabe
        eeat_overall = eeat_analysis.get("scores", {}).get("overall", 0.5)
        if eeat_overall < 0.4:
            additional_issues.append({
                "code": "LOW_EEAT",
                "severity": "WARNING",
                "message": "Niski poziom sygnałów E-E-A-T (ekspertyza, autorytet, wiarygodność)",
                "scores": eeat_analysis.get("scores", {})
            })
        
        # ================================================================
        # 6. POLISH LANGUAGE QUALITY
        # ================================================================
        polish_quality = validate_polish_quality(full_text)
        polish_score = polish_quality.get("score", 70)
        
        # Dodaj rekomendacje językowe
        polish_recs = polish_quality.get("recommendations", [])
        for rec in polish_recs[:2]:
            recommendations.append(f"JĘZYK: {rec}")
        
        # Dodaj issue jeśli jakość językowa słaba
        if polish_score < 50:
            additional_issues.append({
                "code": "LOW_POLISH_QUALITY",
                "severity": "WARNING",
                "message": "Niska jakość języka polskiego - sprawdź kolokacje i powtórzenia",
                "score": polish_score
            })
        
        # ================================================================
        # OBLICZ STATUS I SCORE
        # ================================================================
        errors_count = len(validation_result.get_errors()) + len([i for i in additional_issues if i.get("severity") == "ERROR"])
        warnings_count = len(validation_result.get_warnings()) + len([i for i in additional_issues if i.get("severity") == "WARNING"])
        
        if errors_count > 0:
            status = "WYMAGA_POPRAWEK"
        elif warnings_count > 3:
            status = "WARN"
        else:
            status = "OK"
        
        # Score = średnia ważona: SEO 50%, E-E-A-T 25%, Polish 25%
        base_score = validation_result.score
        base_score -= len([i for i in additional_issues if i.get("severity") == "ERROR"]) * 10
        base_score -= len([i for i in additional_issues if i.get("severity") == "WARNING"]) * 3
        
        final_score = (base_score * 0.50) + (eeat_overall * 100 * 0.25) + (polish_score * 0.25)
        score = max(0, min(100, final_score))
        
        # ================================================================
        # ZAPISZ DO FIRESTORE
        # ================================================================
        review_data = {
            "status": status,
            "score": round(score, 1),
            "word_count": word_count,
            "validation": validation_result.to_dict(),
            "eeat_analysis": eeat_analysis,
            "polish_quality": polish_quality,
            "additional_issues": additional_issues,
            "recommendations": recommendations[:10],
            "summary": {
                "errors": errors_count,
                "warnings": warnings_count,
                "eeat_score": eeat_overall,
                "polish_score": polish_score
            }
        }
        
        doc_ref = db.collection("seo_projects").document(project_id)
        doc_ref.update({
            "final_review": review_data,
            "final_review_timestamp": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "status": status,
            "project_id": project_id,
            "score": score,
            "word_count": word_count,
            "validation": {
                "is_valid": validation_result.is_valid,
                "metrics": validation_result.metrics,
                "keywords": validation_result.keywords_analysis,
                "structure": validation_result.structure_analysis
            },
            "eeat": eeat_analysis,
            "issues": {
                "from_validator": [i.to_dict() for i in validation_result.issues],
                "additional": additional_issues
            },
            "recommendations": recommendations,
            "summary": {
                "errors": errors_count,
                "warnings": warnings_count
            },
            "next_step": "POST /api/project/{id}/apply_corrections" if status != "OK" else "POST /api/project/{id}/paa/analyze"
        }), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[FINAL_REVIEW] ❌ Error: {e}")
        return jsonify({
            "error": str(e),
            "traceback": error_trace
        }), 500


# ================================================================
# 🔧 APPLY CORRECTIONS v23.0
# ================================================================
@final_review_routes.post("/api/project/<project_id>/apply_corrections")
def apply_corrections(project_id):
    """
    Automatyczna korekta artykułu.
    Zapisuje nową wersję z VersionManager.
    """
    try:
        db = firestore.client()
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        batches = data.get("batches", [])
        final_review = data.get("final_review", {})
        main_keyword = data.get("main_keyword", "")
        version_history = data.get("version_history", {})
        
        if not final_review:
            return jsonify({"error": "Run final_review first"}), 400
        
        recommendations = final_review.get("recommendations", [])
        if not recommendations:
            return jsonify({"status": "NO_CORRECTIONS_NEEDED"}), 200
        
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini not configured"}), 500
        
        # Połącz tekst
        full_text = "\n\n".join(b.get("text", "") for b in batches if b.get("text"))
        
        # Przygotuj prompt dla Gemini
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Zbierz brakujące keywords
        keywords_to_add = []
        for issue in final_review.get("additional_issues", []):
            if issue.get("code") == "MISSING_KEYWORD":
                keywords_to_add.append({
                    "keyword": issue["keyword"],
                    "times": issue["missing"]
                })
        
        kw_section = ""
        if keywords_to_add:
            kw_list = "\n".join(f"  - '{k['keyword']}': {k['times']}x" for k in keywords_to_add[:10])
            kw_section = f"FRAZY DO WPLECENIA:\n{kw_list}\n\n"
        
        prompt = f"""Popraw artykuł SEO:

{kw_section}INSTRUKCJE:
{chr(10).join(f"- {r}" for r in recommendations[:8])}

ZASADY:
1. Zachowaj format h2:/h3:
2. Wplataj frazy NATURALNIE
3. "{main_keyword}" częściej niż synonimy
4. Nie zmieniaj struktury drastycznie

ARTYKUŁ:
{full_text[:14000]}

Zwróć TYLKO poprawiony artykuł."""
        
        response = model.generate_content(prompt)
        corrected = response.text.strip()
        corrected = re.sub(r'^```(?:html|markdown)?\n?', '', corrected)
        corrected = re.sub(r'\n?```$', '', corrected)
        
        # Weryfikacja
        verification = {}
        for kw in keywords_to_add[:5]:
            keyword = kw["keyword"].lower()
            before = full_text.lower().count(keyword)
            after = corrected.lower().count(keyword)
            verification[kw["keyword"]] = {
                "before": before,
                "after": after,
                "added": after - before
            }
        
        # Zapisz z wersjonowaniem
        vm = VersionManager.from_dict({"project_id": project_id, "batch_histories": version_history})
        
        # Zapisz jako jeden batch (cały artykuł)
        version = vm.add_version(
            batch_number=0,  # 0 = cały artykuł
            text=corrected,
            source=VersionSource.FINAL_CORRECTIONS,
            metadata={"corrections": recommendations[:5]}
        )
        
        # Aktualizuj Firestore
        doc_ref.update({
            "corrected_article": corrected,
            "corrections_applied": recommendations,
            "correction_verification": verification,
            "correction_timestamp": firestore.SERVER_TIMESTAMP,
            "version_history": vm.to_dict()["batch_histories"]
        })
        
        return jsonify({
            "status": "CORRECTED",
            "version_id": version.version_id,
            "corrections_applied": recommendations,
            "verification": verification,
            "word_count": {
                "before": len(full_text.split()),
                "after": len(corrected.split())
            }
        }), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[FINAL_REVIEW] ❌ Correction error: {e}")
        return jsonify({
            "error": str(e),
            "traceback": error_trace
        }), 500


# ================================================================
# 🔄 ROLLBACK CORRECTIONS
# ================================================================
@final_review_routes.post("/api/project/<project_id>/rollback_corrections")
def rollback_corrections(project_id):
    """Przywraca artykuł sprzed korekty."""
    try:
        db = firestore.client()
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        version_history = data.get("version_history", {})
        
        vm = VersionManager.from_dict({"project_id": project_id, "batch_histories": version_history})
        
        # Znajdź poprzednią wersję (przed FINAL_CORRECTIONS)
        history = vm.get_history(0)  # 0 = cały artykuł
        if not history or len(history.versions) < 2:
            return jsonify({"error": "No previous version to rollback to"}), 400
        
        # Znajdź ostatnią wersję MANUAL
        target_version = None
        for v in reversed(history.versions[:-1]):
            if v.source != VersionSource.FINAL_CORRECTIONS:
                target_version = v
                break
        
        if not target_version:
            return jsonify({"error": "No suitable version found"}), 400
        
        # Rollback
        new_version = vm.rollback_to_version(0, target_version.version_id, "Rollback corrections")
        
        # Usuń corrected_article
        doc_ref.update({
            "corrected_article": firestore.DELETE_FIELD,
            "corrections_applied": firestore.DELETE_FIELD,
            "version_history": vm.to_dict()["batch_histories"]
        })
        
        return jsonify({
            "status": "ROLLED_BACK",
            "rolled_back_to": target_version.version_id,
            "new_version_id": new_version.version_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# HEALTH CHECK
# ================================================================
@final_review_routes.get("/api/final_review/health")
def health_check():
    return jsonify({
        "status": "OK",
        "version": "v23.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "features": ["unified_validator", "version_manager", "rollback"]
    }), 200
