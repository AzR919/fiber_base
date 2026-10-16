"""
Minimal ENCODE REST client and the rule that picks one bigWig per (cell line, assay).

Selection for one cell line + assay:
  1. Candidate experiments: released, human, unperturbed, matching the biosample term name,
     assay title and (for ChIP-seq) target.
  2. Within each experiment: its released bigWigs of the requested assembly and output type;
     keep the one pooled over the most biological replicates (ties: newest file).
  3. Rank experiments: fewest ERROR audits, fewest NOT_COMPLIANT audits, most replicates in the
     chosen file, fewest WARNING audits, newest release, then accession (deterministic).
Audits only rank candidates; they never exclude one (some marks have no audit-clean experiment).
"""

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://www.encodeproject.org"
HUMAN = "Homo sapiens"
AUDIT_LEVELS = ("ERROR", "NOT_COMPLIANT", "WARNING")
FILE_FIELDS = ["accession", "dataset", "biological_replicates", "md5sum", "file_size", "href",
               "date_created", "output_type", "assembly", "status"]


def get_json(path, params=None, retries=4):
    """GET an ENCODE endpoint as JSON. ENCODE answers an empty search with HTTP 404, returned as {"@graph": []}."""
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            time.sleep(0.1)  # stay well under ENCODE's request rate limit
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404 and path.startswith("/search/"):
                return {"@graph": []}
            if attempt == retries - 1 or (e.code < 500 and e.code != 429):
                raise
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)


def search(params):
    query = {"format": "json", "limit": "all"}
    query.update(params)
    return get_json("/search/", query)["@graph"]


def audit_counts(experiment):
    audit = experiment.get("audit") or {}
    return {level: len(audit.get(level, [])) for level in AUDIT_LEVELS}


def _selection(experiment_accession, audit, date_released, f):
    return {
        "file": f["accession"],
        "experiment": experiment_accession,
        "biological_replicates": f.get("biological_replicates") or [],
        "md5sum": f["md5sum"],
        "file_size": f["file_size"],
        "url": BASE_URL + f["href"],
        "audit": audit,
        "date_released": date_released,
    }


def _rank_key(sel):
    a = sel["audit"]
    date = int((sel["date_released"] or "0").replace("-", ""))
    return (a["ERROR"], a["NOT_COMPLIANT"], -len(sel["biological_replicates"]), a["WARNING"], -date, sel["experiment"])


def select_track(term_name, assay_spec, assembly, output_type):
    """Best file for one cell line + assay. Returns (selection dict or None, number of candidate experiments)."""
    exp_params = {
        "type": "Experiment",
        "status": "released",
        "perturbed": "false",
        "replicates.library.biosample.donor.organism.scientific_name": HUMAN,
        "biosample_ontology.term_name": term_name,
        "assay_title": assay_spec["assay_title"],
        "field": ["accession", "audit", "date_released"],
    }
    if assay_spec.get("target"):
        exp_params["target.label"] = assay_spec["target"]

    candidates = []
    for exp in search(exp_params):
        files = search({
            "type": "File",
            "dataset": f"/experiments/{exp['accession']}/",
            "status": "released",
            "file_format": "bigWig",
            "assembly": assembly,
            "output_type": output_type,
            "field": FILE_FIELDS,
        })
        if not files:
            continue
        best = max(files, key=lambda f: (len(f.get("biological_replicates") or []), f.get("date_created", ""), f["accession"]))
        candidates.append(_selection(exp["accession"], audit_counts(exp), exp.get("date_released", ""), best))

    candidates.sort(key=_rank_key)
    return (candidates[0] if candidates else None), len(candidates)


def fetch_locked(entry, assembly, output_type):
    """Re-fetch download info for a file pinned in the lock file, checking it still matches the spec."""
    f = get_json(f"/files/{entry['file']}/", {"format": "json", "frame": "object"})
    if f.get("output_type") != output_type or f.get("assembly") != assembly:
        raise ValueError(
            f"Locked file {entry['file']} is '{f.get('output_type')}' / {f.get('assembly')}, but the spec asks for "
            f"'{output_type}' / {assembly}. Re-run with --reselect to pick new files.")
    if entry.get("md5sum") and entry["md5sum"] != f["md5sum"]:
        raise ValueError(f"Locked file {entry['file']}: md5 in lock ({entry['md5sum']}) != ENCODE ({f['md5sum']}).")
    if f.get("status") != "released":
        print(f"  [WARN] locked file {entry['file']} now has ENCODE status '{f.get('status')}'", flush=True)
    return _selection(entry["experiment"], entry.get("audit", {lvl: 0 for lvl in AUDIT_LEVELS}),
                      entry.get("date_released", ""), f)
