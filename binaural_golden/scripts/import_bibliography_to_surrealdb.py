#!/usr/bin/env python3
"""
Import vibroacoustic bibliography into SurrealDB knowledge base.
Each paper is tagged with semantic domains relevant to Golden Studio project.
"""

import re
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# SurrealDB configuration
SURREAL_URL = "http://localhost:8000/sql"
SURREAL_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "surreal-ns": "research",
    "surreal-db": "knowledge",
    "Authorization": "Basic cm9vdDpyb290"  # root:root base64
}

# Domain mapping based on bibliography sections
SECTION_DOMAINS = {
    "LUTHERIE": ["acoustics", "soundboard", "violin", "guitar", "modal_analysis", "lutherie"],
    "HUMAN BODY": ["body_resonance", "vibration_therapy", "biomechanics", "human_vibration"],
    "VIBROACOUSTIC": ["vibroacoustic_therapy", "VAT", "frequency_therapy", "healing", "clinical"],
    "PLATE VIBRATION": ["plate_physics", "FEM", "modal_analysis", "eigenvalue", "kirchhoff"],
    "EXCITER": ["DML", "exciter_placement", "optimization", "coupling", "transducer"],
    "GOLDEN RATIO": ["golden_ratio", "phi", "sacred_geometry", "fibonacci", "proportions"],
    "CALIBRATION": ["measurement", "calibration", "modal_testing", "FRF", "instrumentation"],
    "WOOD": ["tonewood", "material_properties", "orthotropic", "damping", "acoustics"],
    "TRANSIENT": ["impulse_response", "transient", "attack", "coupling", "phase"],
    "MULTI-EXCITER": ["multi_exciter", "DML", "optimization", "genetic_algorithm", "positioning"],
    "ACOUSTIC BLACK": ["ABH", "acoustic_black_hole", "vibration_isolation", "energy_focusing", "peninsula"],
    "MODE COUPLING": ["modal_coupling", "plate_cavity", "radiation", "topology_optimization"]
}

# Relevance to Golden Studio features
PROJECT_RELEVANCE = {
    "fletcher1998physics": "Foundation for understanding soundboard physics applied to DML plate design",
    "rossing2010science": "Modal analysis techniques used in plate optimization",
    "gough2015violin": "Body modes concept applied to body-zone frequency targeting",
    "schleske2002empirical": "Tap tone analysis informs contact mic calibration approach",
    "schleske2002empirical2": "Psychoacoustic evaluation relevant to ear zone optimization",
    "bissinger2008modal": "FEM analysis methodology applied to plate simulation",
    "woodhouse2014acoustics": "Comprehensive vibration theory underpins physics model",
    "torres2009influence": "Bridge/exciter mass effects on mode shapes",
    "french2008engineering": "Frequency response optimization techniques",
    "caldersmith1995guitar": "Scaling laws for plate dimension optimization",
    "jansson1971topplate": "Mode numbering system used in visualization",
    
    "griffin1990handbook": "CORE: Human body resonance frequencies for zone targeting (spine 10-12Hz, chest 50-60Hz)",
    "randall1997resonance": "Standing body resonances for therapy positioning",
    "paddan1998transmission": "Vibration transfer to head - critical for ear zone design",
    "matsumoto2002effect": "Supine position effects - relevant for vibroacoustic bed",
    "wu1998biodynamic": "Supine position resonances differ from seated - key for table design",
    "fairley1989apparent": "Apparent mass data for body-plate coupling model",
    "kitazaki1998mathematical": "Multi-DOF body model for simulation",
    "manmatha2023chakra": "Chakra frequencies (256-480Hz) used in therapy presets",
    
    "skille1989vibroacoustic": "FOUNDATIONAL: VAT inventor's paper. 30-120Hz sinusoidal - base protocol",
    "wigram1996effects": "Treatment protocols inform therapy program design",
    "campbell2019vibroacoustic": "40Hz stimulation for Parkinson's - specific preset opportunity",
    "lundeberg1991vibratory": "100Hz pain relief - frequency target for therapy",
    "bartel2017music": "40Hz gamma entrainment for Alzheimer's - preset design",
    "naghdi2015effect": "40Hz/23Hz treatment protocols - validated frequencies",
    "punkanen2014contemporary": "Finnish VAT protocols - treatment guidelines",
    
    "leissa1969vibration": "CORE: Analytical plate solutions used in physics model validation",
    "reddy2006theory": "Kirchhoff-Mindlin theory implemented in FEM solver",
    "zienkiewicz2005finite": "FEM fundamentals for scikit-fem implementation",
    "gustafsson2020scikit": "PRIMARY TOOL: scikit-fem library used for modal analysis",
    "mcintyre1983orthotropic": "Wood orthotropic ratios (Ex/Ey ~12:1) for material model",
    "chaigne1994numerical": "Coupling and impedance matching for exciter design",
    
    "harris2010fundamentals": "CORE: DML theory - multiple exciters for uniform response",
    "bank2010modal": "Antinode/node coupling theory for exciter placement",
    "aures2001exciter": "Golden ratio positioning (0.381L, 0.618L) - key algorithm",
    "azizi2015optimization": "GA optimization for exciter placement - validates our approach",
    "wang2020multiple": "Phase control for multiple exciters - timing algorithm",
    
    "livio2003golden": "Golden ratio in music - theoretical foundation",
    "madden1999fibonacci": "Fibonacci in room acoustics and positioning",
    "schroeder1987fractal": "Diffuser design principles",
    "kak2006golden": "Sacred proportions in Indian music - cultural context",
    
    "beranek2012acoustics": "Microphone calibration for contact mic system",
    "ewins2000modal": "Modal testing methodology for plate characterization",
    "richardson1997accelerometer": "Mode shape vs operating shape - analysis approach",
    "avitabile2001experimental": "Practical impact hammer testing - calibration procedure",
    "dpi2020contact": "Contact mic placement techniques",
    "farina2000simultaneous": "Swept-sine calibration method implemented in calibration module",
    
    "bucur2006acoustics": "Wood property database for material selection",
    "obataya2000effects": "Aging effects on wood - material quality assessment",
    "yoshikawa2007acoustical": "Tonewood ranking - material selection guide",
    
    "weinreich1993coupled": "Phase coupling between exciters - timing algorithm basis",
    "woodhouse1998whip": "Whip crack mechanics - sequential impulse timing ('whip effect')",
    "valette1995mechanics": "Transient attack optimization - impulse generation",
    
    "lu2012optimization": "KEY: Multi-exciter + attached masses optimization - Chinese research foundational",
    "lu2009model": "Plate surface separation into exciter regions - zone concept",
    "shen2006positions": "Exciter position effects - optimization targets",
    "zhang2006model": "Attached masses + Rayleigh integral - radiation model",
    "bai2004genetic": "CORE: Genetic algorithm for exciter placement - our NSGA-II basis",
    "jeon2020vibration": "Vibration localization for multi-channel DML - modern approach",
    "pueo2009equalization": "Multi-exciter equalization - DSP filter design",
    "anderson2017optimized": "Array-driven flat panels - mode excitation regions",
    
    "deng2019ring": "KEY: ABH for vibration isolation - SUPPORTS PENINSULA AS BENEFICIAL",
    "krylov2014abh": "FOUNDATIONAL: ABH theory - wave capturing, energy trapping",
    "zhao2014broadband": "ABH for energy harvesting - FE modeling approach",
    "zhao2019abh": "ABH review - peninsula regions as resonators - VALIDATES our scoring",
    "feurtado2017transmission": "Embedded ABH in plates - cutout design",
    "tang2019periodic": "Periodic ABH - band gap generation for isolation",
    "zhao2025cutouts": "ABH with cutouts - SUPPORTS cutout-as-ABH hypothesis in fitness function",
    
    "sum2000modal": "KEY: Modal cross-coupling - peninsula resonance interactions",
    "frendi1994coupling": "Plate-acoustic coupling - radiation model",
    "bokhari2023topology": "TOPOLOGY OPTIMIZATION for loudspeaker - plate shape optimization",
    "chen2021bandgap": "Band gap concept - mode coupling of local resonant modes"
}

def parse_bibtex_entry(entry_text: str) -> Optional[Dict]:
    """Parse a single BibTeX entry into a dictionary."""
    # Extract entry type and key
    match = re.match(r'@(\w+)\{([^,]+),', entry_text)
    if not match:
        return None
    
    entry_type = match.group(1).lower()
    cite_key = match.group(2).strip()
    
    # Extract fields
    fields = {}
    field_pattern = r'(\w+)\s*=\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
    for field_match in re.finditer(field_pattern, entry_text, re.DOTALL):
        field_name = field_match.group(1).lower()
        field_value = field_match.group(2).strip()
        # Clean up LaTeX
        field_value = re.sub(r'\\[\'`^"~]?\{?(\w)\}?', r'\1', field_value)
        field_value = re.sub(r'\{|\}', '', field_value)
        fields[field_name] = field_value
    
    return {
        'type': entry_type,
        'cite_key': cite_key,
        **fields
    }

def determine_section(entry: Dict, all_text: str) -> str:
    """Determine which section the entry belongs to based on context."""
    cite_key = entry['cite_key']
    
    # Find position in file and determine section
    sections = [
        ("LUTHERIE", ["fletcher", "rossing", "gough", "schleske", "bissinger", "woodhouse", "torres", "french", "caldersmith", "jansson"]),
        ("HUMAN BODY", ["griffin", "randall", "paddan", "matsumoto", "wu1998", "fairley", "kitazaki", "manmatha"]),
        ("VIBROACOUSTIC", ["skille", "wigram", "campbell", "lundeberg", "bartel", "naghdi", "punkanen"]),
        ("PLATE VIBRATION", ["leissa", "reddy", "zienkiewicz", "gustafsson", "mcintyre", "chaigne"]),
        ("EXCITER", ["harris", "bank", "aures", "azizi", "wang2020"]),
        ("GOLDEN RATIO", ["livio", "madden", "schroeder", "kak"]),
        ("CALIBRATION", ["beranek", "ewins", "richardson", "avitabile", "dpi", "farina"]),
        ("WOOD", ["bucur", "obataya", "yoshikawa"]),
        ("TRANSIENT", ["weinreich", "woodhouse1998", "valette"]),
        ("MULTI-EXCITER", ["lu2012", "lu2009", "shen2006", "zhang2006", "bai2004", "jeon", "pueo", "anderson"]),
        ("ACOUSTIC BLACK", ["deng", "krylov", "zhao2014", "zhao2019", "feurtado", "tang2019", "zhao2025"]),
        ("MODE COUPLING", ["sum2000", "frendi", "bokhari", "chen2021"])
    ]
    
    for section_name, patterns in sections:
        for pattern in patterns:
            if pattern.lower() in cite_key.lower():
                return section_name
    
    return "GENERAL"

def get_domains(section: str) -> List[str]:
    """Get domain tags based on section."""
    return SECTION_DOMAINS.get(section, ["vibroacoustics", "plate_physics"])

def get_relevance(cite_key: str) -> str:
    """Get project relevance description."""
    return PROJECT_RELEVANCE.get(cite_key, "General reference for vibroacoustic plate design")

def create_paper_record(entry: Dict, section: str) -> Dict:
    """Create a paper record for SurrealDB."""
    cite_key = entry['cite_key']
    
    # Build authors list
    authors = entry.get('author', 'Unknown').split(' and ')
    authors = [a.strip() for a in authors]
    
    # Build abstract/summary from note if available
    note = entry.get('note', '')
    
    # Year
    year = int(entry.get('year', 2000))
    
    # DOI/URL
    doi = entry.get('doi', '')
    url = f"https://doi.org/{doi}" if doi else entry.get('url', '')
    
    return {
        'cite_key': cite_key,
        'title': entry.get('title', 'Untitled'),
        'authors': authors,
        'year': year,
        'type': entry['type'],
        'journal': entry.get('journal', entry.get('booktitle', entry.get('publisher', ''))),
        'volume': entry.get('volume', ''),
        'pages': entry.get('pages', ''),
        'doi': doi,
        'url': url,
        'abstract': note,
        'section': section,
        'domains': get_domains(section),
        'project_relevance': get_relevance(cite_key),
        'imported_at': datetime.now().isoformat(),
        'source': 'vibroacoustic_references.bib',
        'project': 'Golden Studio - Vibroacoustic DML Therapy'
    }

def escape_surreal_string(s: str) -> str:
    """Escape a string for SurrealDB query."""
    if not s:
        return ""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')

def save_to_surrealdb(paper: Dict) -> bool:
    """Save a paper record to SurrealDB."""
    cite_key = paper['cite_key']
    
    # Escape all string fields
    title = escape_surreal_string(paper['title'])
    journal = escape_surreal_string(paper['journal'])
    abstract = escape_surreal_string(paper['abstract'])
    project_relevance = escape_surreal_string(paper['project_relevance'])
    section = paper['section']
    paper_type = paper['type']
    source = paper['source']
    project = paper['project']
    
    # Build the query with proper escaping
    query = f'''
    DELETE paper:{cite_key};
    CREATE paper:{cite_key} SET
        cite_key = "{cite_key}",
        title = "{title}",
        authors = {json.dumps(paper['authors'])},
        year = {paper['year']},
        type = "{paper_type}",
        journal = "{journal}",
        section = "{section}",
        domains = {json.dumps(paper['domains'])},
        project_relevance = "{project_relevance}",
        abstract = "{abstract}",
        source = "{source}",
        project = "{project}"
    ;
    '''
    
    try:
        response = httpx.post(
            SURREAL_URL,
            headers=SURREAL_HEADERS,
            content=query,
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check if CREATE succeeded (second result)
            if len(result) >= 2 and result[1].get('status') == 'OK':
                return True
        
        print(f"     Response: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"  Error saving {cite_key}: {e}")
        return False

def main():
    # Read bibliography file
    bib_path = Path(__file__).parent.parent / "docs" / "research" / "vibroacoustic_references.bib"
    
    if not bib_path.exists():
        print(f"❌ Bibliography not found: {bib_path}")
        return
    
    bib_content = bib_path.read_text()
    
    # Split into entries
    entries = re.split(r'\n(?=@)', bib_content)
    entries = [e.strip() for e in entries if e.strip().startswith('@')]
    
    print(f"📚 Found {len(entries)} bibliography entries")
    print(f"🔗 Connecting to SurrealDB at {SURREAL_URL}...")
    
    # Test connection
    try:
        test_response = httpx.post(
            SURREAL_URL,
            headers=SURREAL_HEADERS,
            content="INFO FOR DB;",
            timeout=5.0
        )
        if test_response.status_code != 200:
            print(f"❌ SurrealDB connection failed: {test_response.status_code}")
            print(f"   Response: {test_response.text}")
            return
        print("✅ SurrealDB connected!")
    except Exception as e:
        print(f"❌ Cannot connect to SurrealDB: {e}")
        print("   Make sure SurrealDB is running: surreal start --user root --pass root")
        return
    
    # Process entries
    success_count = 0
    sections_count = {}
    
    for entry_text in entries:
        entry = parse_bibtex_entry(entry_text)
        if not entry:
            continue
        
        section = determine_section(entry, bib_content)
        sections_count[section] = sections_count.get(section, 0) + 1
        
        paper = create_paper_record(entry, section)
        
        print(f"  📄 {paper['cite_key']}: {paper['title'][:50]}...")
        print(f"     Section: {section}, Domains: {paper['domains'][:3]}")
        
        if save_to_surrealdb(paper):
            success_count += 1
            print(f"     ✅ Saved")
        else:
            print(f"     ⚠️  Save failed (may already exist)")
    
    print(f"\n{'='*60}")
    print(f"📊 IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries: {len(entries)}")
    print(f"Successfully imported: {success_count}")
    print(f"\nBy section:")
    for section, count in sorted(sections_count.items()):
        print(f"  {section}: {count} papers")
    
    print(f"\n🎯 Project: Golden Studio - Vibroacoustic DML Therapy")
    print(f"📖 Each paper tagged with project relevance for semantic queries")

if __name__ == "__main__":
    main()
