"""
FIRESTORE UTILS v1.0
====================
Single Source of Truth dla operacji Firestore.

Zawiera:
- sanitize_for_firestore() - sanityzacja kluczy dla Firestore
- batch_update() - batch write dla wielu operacji

UŻYCIE:
    from firestore_utils import sanitize_for_firestore, batch_update

ZASTĘPUJE duplikaty w:
- project_routes.py
- batch_review_system.py  
- firestore_tracker_routes.py
"""

from typing import Any, Dict, List, Optional
from firebase_admin import firestore


def sanitize_for_firestore(data: Any, depth: int = 0, max_depth: int = 50) -> Any:
    """
    Recursively sanitize dictionary keys for Firestore compatibility.
    
    Firestore restrictions:
    - Keys must be non-empty strings
    - Keys cannot contain: . / [ ] \\ " '
    - Keys cannot start/end with whitespace
    
    Args:
        data: Any data structure (dict, list, or primitive)
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        Sanitized data structure
    """
    if depth > max_depth:
        return data
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip None keys
            if key is None:
                continue
            
            # Convert to string
            str_key = str(key).strip()
            
            # Skip empty keys
            if not str_key:
                continue
            
            # Replace problematic characters for Firestore
            safe_key = (str_key
                .replace('.', '__dot__')
                .replace('/', '__slash__')
                .replace('[', '(')
                .replace(']', ')')
                .replace('\\', '_')
                .replace('"', '')
                .replace("'", '')
            )

            # Ensure key is not empty after sanitization
            if not safe_key:
                safe_key = f"_sanitized_key_{depth}"

            # Handle duplicate keys from collisions
            if safe_key in sanitized:
                safe_key = f'{safe_key}__dup_{depth}'
            
            # Recursively sanitize value
            sanitized[safe_key] = sanitize_for_firestore(value, depth + 1, max_depth)
        
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_for_firestore(item, depth + 1, max_depth) for item in data]
    
    else:
        return data


def batch_update(project_id: str, updates: Dict[str, Any], collection: str = "seo_projects") -> bool:
    """
    Wykonuje batch update na dokumencie Firestore.
    
    Zamiast 3 osobnych doc_ref.update(), wykonuje jeden batch.commit()
    
    Args:
        project_id: ID dokumentu
        updates: Dict z polami do aktualizacji
        collection: Nazwa kolekcji (default: seo_projects)
        
    Returns:
        True jeśli sukces
    """
    try:
        db = firestore.client()
        batch = db.batch()
        
        doc_ref = db.collection(collection).document(project_id)
        
        # Sanitize przed zapisem
        sanitized_updates = sanitize_for_firestore(updates)
        
        batch.update(doc_ref, sanitized_updates)
        batch.commit()
        
        return True
    except Exception as e:
        print(f"[FIRESTORE_UTILS] ❌ batch_update error: {e}")
        return False


def batch_multi_update(operations: List[Dict[str, Any]], collection: str = "seo_projects") -> bool:
    """
    Wykonuje batch update na WIELU dokumentach naraz.
    
    Args:
        operations: Lista dict z {"project_id": ..., "updates": {...}}
        collection: Nazwa kolekcji
        
    Returns:
        True jeśli sukces
    """
    try:
        db = firestore.client()
        batch = db.batch()
        
        for op in operations:
            project_id = op.get("project_id")
            updates = op.get("updates", {})
            
            if not project_id or not updates:
                continue
            
            doc_ref = db.collection(collection).document(project_id)
            sanitized = sanitize_for_firestore(updates)
            batch.update(doc_ref, sanitized)
        
        batch.commit()
        return True
    except Exception as e:
        print(f"[FIRESTORE_UTILS] ❌ batch_multi_update error: {e}")
        return False


def safe_get_project(project_id: str, collection: str = "seo_projects") -> Optional[Dict]:
    """
    Bezpieczne pobranie projektu z Firestore.
    
    Returns:
        Dict z danymi projektu lub None jeśli nie istnieje
    """
    try:
        db = firestore.client()
        doc = db.collection(collection).document(project_id).get()
        
        if not doc.exists:
            return None
        
        return doc.to_dict()
    except Exception as e:
        print(f"[FIRESTORE_UTILS] ❌ safe_get_project error: {e}")
        return None


# ============================================================
# VERSION INFO
# ============================================================

__version__ = "1.0"
__all__ = [
    "sanitize_for_firestore",
    "batch_update",
    "batch_multi_update",
    "safe_get_project",
]
