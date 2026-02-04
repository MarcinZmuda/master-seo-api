"""
===============================================================================
🧪 CLINICALTRIALS.GOV CLIENT v1.0 - REST API v2
===============================================================================
Klient do wyszukiwania badań klinicznych.

API v2 Documentation: https://clinicaltrials.gov/data-api/api
- Darmowe, bez limitu (ale zalecane <10 req/sek)
- Format JSON
- OpenAPI 3.0

Przydatne dla artykułów o:
- Nowych terapiach
- Lekach w fazie badań
- Skuteczności leczenia
===============================================================================
"""

import requests
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class ClinicalTrialsConfig:
    """Konfiguracja klienta ClinicalTrials.gov."""
    
    BASE_URL: str = "https://clinicaltrials.gov/api/v2"
    TIMEOUT: int = 20
    REQUEST_DELAY: float = 0.15  # ~6 req/sek (bezpieczny margines)
    
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Preferowane statusy (ukończone = wyniki dostępne)
    PREFERRED_STATUS: List[str] = field(default_factory=lambda: [
        "COMPLETED",
        "ACTIVE_NOT_RECRUITING",
        "TERMINATED"  # Czasem mają wyniki
    ])
    
    # Preferowane fazy (3 i 4 = najbardziej wiarygodne)
    PREFERRED_PHASES: List[str] = field(default_factory=lambda: [
        "PHASE3",
        "PHASE4"
    ])
    
    # Mapowanie statusów na polski
    STATUS_PL: Dict[str, str] = field(default_factory=lambda: {
        "COMPLETED": "Zakończone",
        "ACTIVE_NOT_RECRUITING": "Aktywne (rekrutacja zakończona)",
        "RECRUITING": "Rekrutacja trwa",
        "NOT_YET_RECRUITING": "Jeszcze nie rozpoczęte",
        "TERMINATED": "Przerwane",
        "SUSPENDED": "Zawieszone",
        "WITHDRAWN": "Wycofane",
        "UNKNOWN": "Status nieznany"
    })
    
    # Mapowanie faz na polski
    PHASE_PL: Dict[str, str] = field(default_factory=lambda: {
        "EARLY_PHASE1": "Faza wczesna 1",
        "PHASE1": "Faza 1",
        "PHASE2": "Faza 2",
        "PHASE3": "Faza 3",
        "PHASE4": "Faza 4 (po rejestracji)",
        "NA": "Nie dotyczy"
    })


CONFIG = ClinicalTrialsConfig()


# ============================================================================
# KLIENT CLINICALTRIALS.GOV
# ============================================================================

class ClinicalTrialsClient:
    """Klient ClinicalTrials.gov REST API v2."""
    
    def __init__(self, config: ClinicalTrialsConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BRAJEN-SEO-Medical/1.0"
        })
        self._last_request_time = 0
        
        print("[CLINICALTRIALS] ✅ Client initialized (API v2)")
    
    def _rate_limit(self):
        """Rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.REQUEST_DELAY:
            time.sleep(self.config.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    # ========================================================================
    # WYSZUKIWANIE
    # ========================================================================
    
    def search_studies(
        self,
        condition: str = None,
        intervention: str = None,
        term: str = None,
        max_results: int = 20,
        status: List[str] = None,
        phase: List[str] = None,
        has_results: bool = None,
        country: str = None
    ) -> Dict[str, Any]:
        """
        Wyszukuje badania kliniczne.
        
        Args:
            condition: Choroba/stan (np. "diabetes", "breast cancer")
            intervention: Interwencja (np. "metformin", "surgery")
            term: Ogólne wyszukiwanie (jeśli nie podano condition/intervention)
            max_results: Maksymalna liczba wyników
            status: Filtry statusu ["COMPLETED", "RECRUITING", etc.]
            phase: Filtry fazy ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
            has_results: Tylko badania z wynikami (True) lub bez (False)
            country: Kraj (np. "Poland", "United States")
        
        Returns:
            {
                "status": "OK",
                "total_count": 500,
                "studies": [...]
            }
        
        Example:
            >>> client.search_studies(
            ...     condition="type 2 diabetes",
            ...     intervention="metformin",
            ...     status=["COMPLETED"],
            ...     phase=["PHASE3", "PHASE4"]
            ... )
        """
        self._rate_limit()
        
        # Buduj parametry zapytania
        params = {
            "pageSize": min(max_results, self.config.MAX_PAGE_SIZE),
            "sort": "LastUpdatePostDate:desc"
        }
        
        # Buduj query
        query_parts = []
        if condition:
            query_parts.append(f"AREA[Condition]{condition}")
        if intervention:
            query_parts.append(f"AREA[Intervention]{intervention}")
        if term:
            query_parts.append(term)
        
        if query_parts:
            params["query.term"] = " AND ".join(query_parts)
        elif condition:
            params["query.cond"] = condition
        
        # Filtry
        if status:
            params["filter.overallStatus"] = ",".join(status)
        if phase:
            params["filter.phase"] = ",".join(phase)
        if has_results is not None:
            params["filter.resultsFirstSubmitDate"] = "MIN" if has_results else ""
        if country:
            params["query.locn"] = country
        
        try:
            url = f"{self.config.BASE_URL}/studies"
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            # Formatuj wyniki
            formatted_studies = [self._format_study(s) for s in studies]
            formatted_studies = [s for s in formatted_studies if s]  # Usuń None
            
            return {
                "status": "OK",
                "condition": condition,
                "intervention": intervention,
                "total_count": data.get("totalCount", len(studies)),
                "returned": len(formatted_studies),
                "studies": formatted_studies
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[CLINICALTRIALS] ❌ Request error: {e}")
            return {"status": "ERROR", "error": str(e), "studies": []}
        except Exception as e:
            print(f"[CLINICALTRIALS] ❌ Error: {e}")
            return {"status": "ERROR", "error": str(e), "studies": []}
    
    def get_study(self, nct_id: str) -> Optional[Dict]:
        """
        Pobiera szczegóły pojedynczego badania po NCT ID.
        
        Args:
            nct_id: Identyfikator badania (np. "NCT04267848")
        
        Returns:
            Dict z danymi badania lub None
        """
        self._rate_limit()
        
        try:
            url = f"{self.config.BASE_URL}/studies/{nct_id}"
            response = self.session.get(url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            return self._format_study(response.json())
            
        except Exception as e:
            print(f"[CLINICALTRIALS] ❌ Get study error: {e}")
            return None
    
    def _format_study(self, study_data: Dict) -> Optional[Dict]:
        """Formatuje dane badania do standardowego formatu."""
        
        try:
            protocol = study_data.get("protocolSection", {})
            
            # ================================================================
            # IDENTYFIKACJA
            # ================================================================
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId", "")
            org_study_id = id_module.get("orgStudyIdInfo", {}).get("id", "")
            
            # Tytuł (oficjalny lub krótki)
            title = id_module.get("officialTitle") or id_module.get("briefTitle", "")
            brief_title = id_module.get("briefTitle", "")
            
            # ================================================================
            # STATUS
            # ================================================================
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus", "UNKNOWN")
            status_pl = self.config.STATUS_PL.get(overall_status, overall_status)
            
            # Daty
            start_date = self._parse_date(status_module.get("startDateStruct", {}))
            completion_date = self._parse_date(status_module.get("completionDateStruct", {}))
            first_posted = self._parse_date(status_module.get("studyFirstPostDateStruct", {}))
            last_update = self._parse_date(status_module.get("lastUpdatePostDateStruct", {}))
            
            # ================================================================
            # DESIGN
            # ================================================================
            design_module = protocol.get("designModule", {})
            study_type = design_module.get("studyType", "")
            
            phases = design_module.get("phases", [])
            phases_pl = [self.config.PHASE_PL.get(p, p) for p in phases]
            
            # Enrollment
            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count", 0)
            enrollment_type = enrollment_info.get("type", "")
            
            # ================================================================
            # OPIS
            # ================================================================
            desc_module = protocol.get("descriptionModule", {})
            brief_summary = desc_module.get("briefSummary", "")
            detailed_desc = desc_module.get("detailedDescription", "")
            
            # ================================================================
            # WARUNKI I INTERWENCJE
            # ================================================================
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            keywords = conditions_module.get("keywords", [])
            
            # Interwencje
            arms_module = protocol.get("armsInterventionsModule", {})
            interventions = []
            for interv in arms_module.get("interventions", []):
                interventions.append({
                    "type": interv.get("type", ""),
                    "name": interv.get("name", ""),
                    "description": interv.get("description", "")[:200] if interv.get("description") else ""
                })
            
            # ================================================================
            # SPONSOR
            # ================================================================
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name", "")
            sponsor_class = lead_sponsor.get("class", "")  # INDUSTRY, NIH, etc.
            
            collaborators = [c.get("name", "") for c in sponsor_module.get("collaborators", [])]
            
            # ================================================================
            # WYNIKI (jeśli dostępne)
            # ================================================================
            has_results = "resultsSection" in study_data
            results_summary = ""
            if has_results:
                results = study_data.get("resultsSection", {})
                # Pobierz primary outcome
                outcomes = results.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
                if outcomes:
                    primary = outcomes[0]
                    results_summary = primary.get("title", "")
            
            # ================================================================
            # LOKALIZACJE
            # ================================================================
            locations_module = protocol.get("contactsLocationsModule", {})
            locations = []
            for loc in locations_module.get("locations", [])[:5]:  # Max 5
                locations.append({
                    "facility": loc.get("facility", ""),
                    "city": loc.get("city", ""),
                    "country": loc.get("country", "")
                })
            
            # Czy w Polsce?
            in_poland = any(loc.get("country", "").lower() == "poland" for loc in locations_module.get("locations", []))
            
            return {
                "nct_id": nct_id,
                "org_study_id": org_study_id,
                "title": title,
                "brief_title": brief_title,
                "status": overall_status,
                "status_pl": status_pl,
                "phases": phases,
                "phases_pl": phases_pl,
                "study_type": study_type,
                "brief_summary": brief_summary[:500] + "..." if len(brief_summary) > 500 else brief_summary,
                "conditions": conditions,
                "keywords": keywords[:10],
                "interventions": interventions[:5],
                "lead_sponsor": sponsor_name,
                "sponsor_class": sponsor_class,
                "collaborators": collaborators[:3],
                "enrollment": enrollment,
                "enrollment_type": enrollment_type,
                "start_date": start_date,
                "completion_date": completion_date,
                "first_posted": first_posted,
                "last_update": last_update,
                "has_results": has_results,
                "results_summary": results_summary,
                "locations": locations,
                "in_poland": in_poland,
                "source": "ClinicalTrials.gov",
                "url": f"https://clinicaltrials.gov/study/{nct_id}"
            }
            
        except Exception as e:
            print(f"[CLINICALTRIALS] ⚠️ Format error: {e}")
            return None
    
    def _parse_date(self, date_struct: Dict) -> str:
        """Parsuje strukturę daty z API."""
        if not date_struct:
            return ""
        
        date_str = date_struct.get("date", "")
        # Format: "2023-05-15" lub "May 2023"
        return date_str
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def search_completed_trials(
        self,
        condition: str,
        intervention: str = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Wyszukuje ukończone badania (z wynikami).
        Idealne do cytowania w artykułach.
        """
        return self.search_studies(
            condition=condition,
            intervention=intervention,
            max_results=max_results,
            status=["COMPLETED"],
            phase=["PHASE3", "PHASE4"]
        )
    
    def search_polish_trials(
        self,
        condition: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Wyszukuje badania prowadzone w Polsce."""
        return self.search_studies(
            condition=condition,
            max_results=max_results,
            country="Poland"
        )


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_client: Optional[ClinicalTrialsClient] = None


def get_clinicaltrials_client() -> ClinicalTrialsClient:
    """Zwraca singleton klienta."""
    global _client
    if _client is None:
        _client = ClinicalTrialsClient()
    return _client


def search_clinical_trials(
    condition: str,
    intervention: str = None,
    max_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Główna funkcja do wyszukiwania badań klinicznych.
    
    Example:
        >>> result = search_clinical_trials("diabetes", "metformin", max_results=5)
        >>> for study in result["studies"]:
        ...     print(f"{study['nct_id']}: {study['brief_title']}")
    """
    return get_clinicaltrials_client().search_studies(
        condition=condition,
        intervention=intervention,
        max_results=max_results,
        **kwargs
    )


def search_completed_trials(
    condition: str,
    intervention: str = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """Wyszukuje ukończone badania fazy 3/4."""
    return get_clinicaltrials_client().search_completed_trials(
        condition=condition,
        intervention=intervention,
        max_results=max_results
    )


# ============================================================================
# EXPORT
# ============================================================================

CLINICALTRIALS_AVAILABLE = True

__all__ = [
    "ClinicalTrialsClient",
    "ClinicalTrialsConfig",
    "CONFIG",
    "get_clinicaltrials_client",
    "search_clinical_trials",
    "search_completed_trials",
    "CLINICALTRIALS_AVAILABLE"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 CLINICALTRIALS.GOV CLIENT v1.0 TEST")
    print("=" * 60)
    
    client = get_clinicaltrials_client()
    
    # Test wyszukiwania
    print("\n📋 Test: Ukończone badania 'diabetes metformin'...")
    result = client.search_completed_trials(
        condition="type 2 diabetes",
        intervention="metformin",
        max_results=3
    )
    
    print(f"Status: {result['status']}")
    print(f"Total found: {result.get('total_count', 0)}")
    print(f"Returned: {result.get('returned', 0)}")
    
    for study in result.get("studies", [])[:3]:
        print(f"\n🔬 {study['nct_id']}")
        print(f"   Title: {study['brief_title'][:70]}...")
        print(f"   Status: {study['status_pl']}")
        print(f"   Phases: {', '.join(study['phases_pl'])}")
        print(f"   Sponsor: {study['lead_sponsor']}")
        print(f"   Enrollment: {study['enrollment']}")
        print(f"   Has results: {'✅' if study['has_results'] else '❌'}")
