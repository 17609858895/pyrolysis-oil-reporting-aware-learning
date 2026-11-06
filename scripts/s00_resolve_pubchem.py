"""Resolve compound names (name_verbatim) to PubChem structures (CID, SMILES, InChIKey, formula).
Read-only public lookups, rate-limited (<=4 req/s), cached to cache/pubchem_names.csv so it can resume."""
import re, time, json, csv, os, sys, urllib.parse, urllib.request
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('PYROLYSIS_DATA', os.path.join(ROOT, 'data', 'compound_records.xlsx'))
CACHE = os.path.join(ROOT, 'cache', 'pubchem_names.csv')

def candidates(name):
    n = str(name).replace('\n', ' ').strip()
    out = [n]
    # "A (B)" -> try A and B
    m = re.match(r'^(.*?)\s*\((.*?)\)\s*$', n)
    if m:
        out += [m.group(1).strip(), m.group(2).strip()]
    # remove trailing stereo / isomer notes
    out.append(re.sub(r'\s*\[.*?isomer.*?\]', '', n, flags=re.I))
    out.append(n.replace('–', '-').replace('—', '-').replace('’', "'"))
    seen, res = set(), []
    for c in out:
        c = c.strip(' ,;')
        if c and c.lower() not in seen and len(c) > 1:
            seen.add(c.lower()); res.append(c)
    return res

def query(name):
    url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + urllib.parse.quote(name, safe='') +
           '/property/MolecularFormula,CanonicalSMILES,IsomericSMILES,InChIKey,MolecularWeight/JSON')
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            js = json.loads(r.read().decode())
        p = js['PropertyTable']['Properties'][0]
        return p
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        if e.code == 503: time.sleep(2); return 'retry'
        return None
    except Exception:
        return 'retry'

def main():
    df = pd.read_excel(DATA, 'Extracted data (8738)')
    df['nm'] = df['name_verbatim'].astype(str).str.strip()
    freq = df.groupby('nm').record_id.nunique().sort_values(ascending=False)
    names = list(freq.index)                       # most widely reported names first
    done = {}
    if os.path.exists(CACHE):
        for r in csv.DictReader(open(CACHE, encoding='utf-8')):
            done[r['name']] = r
    f = open(CACHE, 'a', encoding='utf-8', newline='')
    w = csv.DictWriter(f, fieldnames=['name', 'query_used', 'cid', 'formula', 'smiles', 'iso_smiles', 'inchikey', 'mw'])
    if not done: w.writeheader()
    todo = [n for n in names if n not in done]
    print('names', len(names), 'todo', len(todo), flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    def resolve(n):
        rec = {'name': n, 'query_used': '', 'cid': '', 'formula': '', 'smiles': '', 'iso_smiles': '', 'inchikey': '', 'mw': ''}
        for c in candidates(n):
            for attempt in range(3):
                p = query(c); time.sleep(1.0)   # 4 workers x ~1 req/s  <= PubChem 5 req/s policy
                if p != 'retry': break
            if isinstance(p, dict):
                rec.update(query_used=c, cid=p.get('CID', ''), formula=p.get('MolecularFormula', ''),
                           smiles=p.get('CanonicalSMILES', p.get('ConnectivitySMILES', '')),
                           iso_smiles=p.get('IsomericSMILES', p.get('SMILES', '')),
                           inchikey=p.get('InChIKey', ''), mw=p.get('MolecularWeight', ''))
                break
        with lock:
            w.writerow(rec); f.flush()
        return n
    with ThreadPoolExecutor(4) as ex:
        for i, n in enumerate(ex.map(resolve, todo)):
            if i % 200 == 0: print(i, n, flush=True)
    f.close(); print('done', flush=True)

if __name__ == '__main__':
    main()
