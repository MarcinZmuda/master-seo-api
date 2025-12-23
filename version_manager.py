"""
===============================================================================
📚 VERSION MANAGER v23.0 - Wersjonowanie treści z możliwością rollbacku
===============================================================================
Rozwiązuje PROBLEM 5: Brak Wersjonowania Treści

Każda zmiana tekstu jest zapisywana jako nowa wersja.
Możliwość rollbacku do poprzedniej wersji.

===============================================================================
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class VersionSource(Enum):
    """Źródło wersji."""
    MANUAL = "manual"           # Ręcznie napisane przez GPT
    AUTO_CORRECT = "auto_correct"  # Auto-korekta keywords
    FINAL_CORRECTIONS = "final_corrections"  # Korekty z final review
    ROLLBACK = "rollback"       # Rollback do poprzedniej wersji
    IMPORT = "import"           # Import zewnętrzny


@dataclass
class ContentVersion:
    """Pojedyncza wersja treści."""
    version_id: str
    version_number: int
    text: str
    text_hash: str
    source: VersionSource
    created_at: str
    parent_version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    is_current: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "text": self.text,
            "text_hash": self.text_hash,
            "source": self.source.value,
            "created_at": self.created_at,
            "parent_version_id": self.parent_version_id,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "is_current": self.is_current
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContentVersion':
        return cls(
            version_id=data.get("version_id", ""),
            version_number=data.get("version_number", 1),
            text=data.get("text", ""),
            text_hash=data.get("text_hash", ""),
            source=VersionSource(data.get("source", "manual")),
            created_at=data.get("created_at", ""),
            parent_version_id=data.get("parent_version_id"),
            metadata=data.get("metadata", {}),
            word_count=data.get("word_count", 0),
            is_current=data.get("is_current", False)
        )


@dataclass
class BatchVersionHistory:
    """Historia wersji dla jednego batcha."""
    batch_number: int
    versions: List[ContentVersion] = field(default_factory=list)
    current_version_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "batch_number": self.batch_number,
            "versions": [v.to_dict() for v in self.versions],
            "current_version_id": self.current_version_id,
            "version_count": len(self.versions)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BatchVersionHistory':
        versions = [ContentVersion.from_dict(v) for v in data.get("versions", [])]
        return cls(
            batch_number=data.get("batch_number", 0),
            versions=versions,
            current_version_id=data.get("current_version_id")
        )


def generate_version_id() -> str:
    """Generuje unikalny ID wersji."""
    import uuid
    return f"v_{uuid.uuid4().hex[:12]}"


def calculate_text_hash(text: str) -> str:
    """Oblicza hash tekstu (do wykrywania duplikatów)."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def create_version(
    text: str,
    source: VersionSource,
    parent_version_id: Optional[str] = None,
    metadata: Dict = None,
    version_number: int = 1
) -> ContentVersion:
    """
    Tworzy nową wersję treści.
    """
    return ContentVersion(
        version_id=generate_version_id(),
        version_number=version_number,
        text=text,
        text_hash=calculate_text_hash(text),
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_version_id=parent_version_id,
        metadata=metadata or {},
        word_count=len(text.split()),
        is_current=True
    )


class VersionManager:
    """
    Zarządza wersjami treści dla projektu.
    """
    
    def __init__(self, project_id: str, batch_histories: Dict[int, BatchVersionHistory] = None):
        self.project_id = project_id
        self.batch_histories: Dict[int, BatchVersionHistory] = batch_histories or {}
    
    def add_version(
        self,
        batch_number: int,
        text: str,
        source: VersionSource,
        metadata: Dict = None
    ) -> ContentVersion:
        """
        Dodaje nową wersję dla batcha.
        """
        # Pobierz lub utwórz historię dla batcha
        if batch_number not in self.batch_histories:
            self.batch_histories[batch_number] = BatchVersionHistory(batch_number=batch_number)
        
        history = self.batch_histories[batch_number]
        
        # Sprawdź czy tekst się zmienił
        new_hash = calculate_text_hash(text)
        if history.versions:
            current = self.get_current_version(batch_number)
            if current and current.text_hash == new_hash:
                return current
        
        # Oznacz poprzednie wersje jako nieaktualne
        for v in history.versions:
            v.is_current = False
        
        # Utwórz nową wersję
        parent_id = history.current_version_id
        version_number = len(history.versions) + 1
        
        new_version = create_version(
            text=text,
            source=source,
            parent_version_id=parent_id,
            metadata=metadata,
            version_number=version_number
        )
        
        history.versions.append(new_version)
        history.current_version_id = new_version.version_id
        
        return new_version
    
    def get_current_version(self, batch_number: int) -> Optional[ContentVersion]:
        """Zwraca aktualną wersję batcha."""
        if batch_number not in self.batch_histories:
            return None
        
        history = self.batch_histories[batch_number]
        for v in reversed(history.versions):
            if v.is_current:
                return v
        
        return history.versions[-1] if history.versions else None
    
    def get_version_by_id(self, batch_number: int, version_id: str) -> Optional[ContentVersion]:
        """Zwraca konkretną wersję po ID."""
        if batch_number not in self.batch_histories:
            return None
        
        for v in self.batch_histories[batch_number].versions:
            if v.version_id == version_id:
                return v
        
        return None
    
    def get_version_by_number(self, batch_number: int, version_number: int) -> Optional[ContentVersion]:
        """Zwraca konkretną wersję po numerze."""
        if batch_number not in self.batch_histories:
            return None
        
        for v in self.batch_histories[batch_number].versions:
            if v.version_number == version_number:
                return v
        
        return None
    
    def rollback_to_version(
        self,
        batch_number: int,
        version_id: str,
        reason: str = ""
    ) -> Optional[ContentVersion]:
        """
        Przywraca poprzednią wersję.
        Tworzy NOWĄ wersję z treścią starej (nie usuwa historii).
        """
        target_version = self.get_version_by_id(batch_number, version_id)
        if not target_version:
            return None
        
        metadata = {
            "rollback_from": self.batch_histories[batch_number].current_version_id,
            "rollback_to": version_id,
            "reason": reason
        }
        
        return self.add_version(
            batch_number=batch_number,
            text=target_version.text,
            source=VersionSource.ROLLBACK,
            metadata=metadata
        )
    
    def get_history(self, batch_number: int) -> Optional[BatchVersionHistory]:
        """Zwraca pełną historię batcha."""
        return self.batch_histories.get(batch_number)
    
    def get_all_histories(self) -> Dict[int, BatchVersionHistory]:
        """Zwraca wszystkie historie."""
        return self.batch_histories
    
    def compare_versions(
        self,
        batch_number: int,
        version_id_1: str,
        version_id_2: str
    ) -> Dict:
        """
        Porównuje dwie wersje.
        """
        v1 = self.get_version_by_id(batch_number, version_id_1)
        v2 = self.get_version_by_id(batch_number, version_id_2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        return {
            "version_1": {
                "id": v1.version_id,
                "number": v1.version_number,
                "word_count": v1.word_count,
                "created_at": v1.created_at,
                "source": v1.source.value
            },
            "version_2": {
                "id": v2.version_id,
                "number": v2.version_number,
                "word_count": v2.word_count,
                "created_at": v2.created_at,
                "source": v2.source.value
            },
            "word_count_diff": v2.word_count - v1.word_count,
            "same_content": v1.text_hash == v2.text_hash
        }
    
    def to_dict(self) -> Dict:
        """Serializuje do dict (do zapisu w Firestore)."""
        return {
            "project_id": self.project_id,
            "batch_histories": {
                str(k): v.to_dict() for k, v in self.batch_histories.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionManager':
        """Deserializuje z dict."""
        batch_histories = {}
        for k, v in data.get("batch_histories", {}).items():
            batch_histories[int(k)] = BatchVersionHistory.from_dict(v)
        
        return cls(
            project_id=data.get("project_id", ""),
            batch_histories=batch_histories
        )


# ================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================
def get_version_diff_summary(v1: ContentVersion, v2: ContentVersion) -> str:
    """
    Generuje tekstowe podsumowanie różnic między wersjami.
    """
    lines = []
    lines.append(f"Wersja {v1.version_number} → {v2.version_number}")
    lines.append(f"Źródło zmiany: {v2.source.value}")
    lines.append(f"Zmiana słów: {v1.word_count} → {v2.word_count} ({v2.word_count - v1.word_count:+d})")
    
    if v2.metadata.get("reason"):
        lines.append(f"Powód: {v2.metadata['reason']}")
    
    return "\n".join(lines)


def create_version_manager_for_project(project_id: str, existing_batches: List[Dict] = None) -> VersionManager:
    """
    Tworzy VersionManager dla projektu, opcjonalnie importując istniejące batche.
    """
    vm = VersionManager(project_id)
    
    if existing_batches:
        for i, batch in enumerate(existing_batches):
            batch_num = i + 1
            text = batch.get("text", "")
            if text:
                vm.add_version(
                    batch_number=batch_num,
                    text=text,
                    source=VersionSource.IMPORT,
                    metadata={"imported_from": "existing_batch"}
                )
    
    return vm
