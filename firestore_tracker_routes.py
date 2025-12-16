@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    Zapisuje batch i automatycznie uruchamia końcowy audyt (Gemini),
    jeśli to był ostatni batch.
    """
    data = request.get_json(force=True)
    text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

    # Zapisz batch
    result = process_batch_in_firestore(project_id, text, meta_trace, forced)
    result["mode"] = "APPROVE"

    # 🔹 Sprawdź, czy to ostatni batch (POPRAWIONA LOGIKA)
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if doc.exists:
        project_data = doc.to_dict()
        # Używamy total_planned_batches, które ustawiliśmy przy tworzeniu projektu
        total_planned = project_data.get("total_planned_batches", 0)
        current_batches = project_data.get("batches", [])
        total_current = len(current_batches)

        print(f"[TRACKER] Batch check: {total_current}/{total_planned}")

        # Trigger działa, gdy mamy klucz API i osiągnęliśmy limit batchy
        if GEMINI_API_KEY and total_planned > 0 and total_current >= total_planned:
            
            existing_fr = project_data.get("final_review")
            # Nie generuj ponownie, chyba że wymuszono
            if existing_fr and not forced:
                result["final_review_status"] = "EXISTS"
                result["next_action"] = "Final review już istnieje."
                return jsonify(result), 200
            
            try:
                print(f"[TRACKER] 🏁 Final batch detected ({total_current}/{total_planned}) → Running Gemini Review...")
                
                # Dynamiczny import żeby uniknąć cyklicznych zależności
                from final_review_routes import perform_final_review_logic
                
                # Wywołanie logiki review
                review_result = perform_final_review_logic(project_id, project_data, current_batches)
                
                result["final_review"] = review_result.get("review_text")
                result["final_review_status"] = "GENERATED"
                result["article_length"] = review_result.get("article_length")
                result["next_action"] = "Zastosuj poprawki: POST /api/project/<id>/apply_final_corrections"
                
            except Exception as e:
                print(f"[TRACKER] ❌ Błąd triggera Final Review: {e}")
                result["final_review_error"] = str(e)

    return jsonify(result), 200
