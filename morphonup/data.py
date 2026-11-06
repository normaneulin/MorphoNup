"""Load 147-bp DatasetNup exports with ID, coordinate, RC and feature audits."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

BP=["Buckle","EP","MGW","Opening","ProT","Shear","Stagger","Stretch"]
GEOMETRY_BP=[n for n in BP if n!="EP"]
STEP=["HelT","Rise","Roll","Shift","Slide","Tilt"]
DESCRIPTOR_NAMES=BP+STEP
SIGNS=np.array([-1 if n in {"Buckle","Shear","Shift","Tilt"} else 1 for n in DESCRIPTOR_NAMES],dtype=np.float32)[None,None,:]

def load_species(folder, invalid_policy="exclude"):
    DATA_DIR=Path(folder)
    files=sorted((DATA_DIR/"sequence").glob("*.fas"))
    assert len(files)==1, files
    FASTA_NAME=files[0].name
    def read_records(path):
        ids, values = [], []
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    ids.append(line[1:]); values.append([])
                else:
                    if not values:
                        raise ValueError(f"Content before first header: {path}")
                    values[-1].append(line)
        return ids, values
    
    fasta = DATA_DIR/"sequence"/FASTA_NAME
    ids, chunks = read_records(fasta)
    seqs = ["".join(v).upper() for v in chunks]
    assert len(ids) == len(set(ids))
    source_ids=list(ids);source_n=len(ids)
    keep=np.array([len(s)==147 and set(s)<=set("ACGT") for s in seqs],dtype=bool)
    excluded=[dict(source_index=i,sample_id=ids[i],length=len(seq),
                   reason="non_147bp" if len(seq)!=147 else "ambiguous_or_invalid_bases",
                   invalid_bases="".join(sorted(set(seq)-set("ACGT"))))
              for i,seq in enumerate(seqs) if not keep[i]]
    if invalid_policy not in ("exclude","error"):raise ValueError("invalid_policy must be exclude or error")
    if excluded and invalid_policy=="error":raise ValueError(f"{len(excluded)} invalid sequences; first: {excluded[0]}")
    if not keep.any():raise ValueError("No valid 147-bp A/C/G/T sequences remain.")
    source_indices=np.flatnonzero(keep)
    ids=[ids[i] for i in source_indices];seqs=[seqs[i] for i in source_indices]
    if excluded:print(f"Predeclared QC: excluded {len(excluded)} / {source_n} ambiguous/non-147-bp records; same cohort for every modality.",flush=True)
    assert all(i.startswith(("nucleosomal_sequence","linker_sequence")) for i in ids)
    y = np.array([int(i.startswith("nucleosomal_sequence")) for i in ids], dtype=np.int32)
    enc = np.array([["ACGT".index(c) for c in s] for s in seqs], dtype=np.int32)
    X = np.eye(4, dtype=np.float32)[enc]
    Xrc = X[:,::-1,::-1].copy()
    enc_rc = 3-enc[:,::-1]
    raw, paths = {}, [fasta]
    for name in DESCRIPTOR_NAMES:
        path = DATA_DIR/"shape"/(FASTA_NAME+"."+name)
        sid, rows = read_records(path)
        assert sid == source_ids, f"Shape ID/order mismatch: {name}"
        arr = np.stack([np.fromstring(
            ",".join(row).replace("NA","nan"), sep=",", dtype=np.float32)
            for row,valid in zip(rows,keep) if valid])
        width = 147 if name in BP else 146
        assert arr.shape == (len(y),width), (name,arr.shape)
        mask = np.zeros(width,dtype=bool)
        mask[[0,1,145,146] if name in BP else [0,145]] = True
        assert np.array_equal(np.isnan(arr),np.broadcast_to(mask,arr.shape)), name
        raw[name] = arr; paths.append(path)
    
    def centered(native):
        return np.stack([native[n][:,2:145] if n in BP else
                         0.5*(native[n][:,1:144]+native[n][:,2:145])
                         for n in DESCRIPTOR_NAMES],axis=-1).astype(np.float32)
    
    H = centered(raw)
    native_rc = {n:raw[n][:,::-1] * float(SIGNS[0,0,j])
                 for j,n in enumerate(DESCRIPTOR_NAMES)}
    Hrc = centered(native_rc)
    np.testing.assert_allclose(Hrc,H[:,::-1,:]*SIGNS,atol=1e-6,rtol=1e-6)
    assert H.shape == (len(y),143,14) and np.isfinite(H).all()
    # Sequence-based BP reference validates channel signs independently of array reversal.
    def kmer_codes(e,k):
        out = np.zeros((len(e),e.shape[1]-k+1),dtype=np.int32)
        for offset in range(k):
            out = 4*out + e[:,offset:offset+out.shape[1]]
        return out
    p5, r5 = kmer_codes(enc,5), kmer_codes(enc_rc,5)
    for j,n in enumerate(BP):
        lut = np.full(1024,np.nan,dtype=np.float32)
        lut[p5.ravel()] = H[:,:,j].ravel()
        np.testing.assert_allclose(lut[p5],H[:,:,j],atol=1e-6)
        known=np.isfinite(lut[r5])
        np.testing.assert_allclose(lut[r5][known],Hrc[:,:,j][known],atol=1e-6)
    
    # Interior step values use the surrounding hexamer; validate their RC signs independently.
    p6, r6 = kmer_codes(enc,6), kmer_codes(enc_rc,6)
    for n in STEP:
        lut = np.full(4096,np.nan,dtype=np.float32)
        lut[p6.ravel()] = raw[n][:,2:144].ravel()
        np.testing.assert_allclose(lut[p6],raw[n][:,2:144],atol=1e-6)
        known=np.isfinite(lut[r6])
        np.testing.assert_allclose(lut[r6][known],native_rc[n][:,2:144][known],atol=1e-6)
    
    def kmer_features(e):
        features=[]
        for k in range(1,5):
            codes=kmer_codes(e,k)
            features.append(np.stack([np.bincount(row,minlength=4**k)/len(row)
                                      for row in codes]).astype(np.float32))
        return np.concatenate(features,axis=1)
    K, Krc = kmer_features(enc), kmer_features(enc_rc)
    assert K.shape == (len(y),340)
    canon = [min(s,s.translate(str.maketrans("ACGT","TGCA"))[::-1]) for s in seqs]
    GROUP_IDS,group_keys=pd.factorize(np.asarray(canon,dtype=object),sort=True)
    DUPLICATE_EXCESS=len(canon)-len(group_keys)
    group_label_counts=pd.DataFrame({"group":GROUP_IDS,"label":y}).groupby("group").label.nunique()
    conflicts=group_label_counts[group_label_counts>1].index.to_numpy()
    SPLIT_STRATEGY="stratified_kfold_on_canonical_groups" if DUPLICATE_EXCESS else "stratified_kfold"
    hashes = {str(p.relative_to(DATA_DIR)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    manifest=dict(files=hashes,n=len(y),source_n=source_n,excluded_n=len(excluded),invalid_sequence_policy=invalid_policy,positive=int(y.sum()),
        negative=int((1-y).sum()),descriptor_names=DESCRIPTOR_NAMES,
        canonical_groups=len(group_keys),duplicate_or_rc_excess=DUPLICATE_EXCESS,
        split_strategy=SPLIT_STRATEGY,
        representation="native geometric BP 143x7, step 144x6, separate electrostatic EP 143x1",
        dnashaper_version="not recorded in supplied exports")
    # Native valid coordinates: BP positions 2..144; step indices 1..144.
    B = np.stack([raw[n][:,2:145] for n in BP], axis=-1)
    T = np.stack([raw[n][:,1:145] for n in STEP], axis=-1)
    Brc = np.stack([native_rc[n][:,2:145] for n in BP], axis=-1)
    Trc = np.stack([native_rc[n][:,1:145] for n in STEP], axis=-1)
    np.testing.assert_allclose(Brc, B[:,::-1]*SIGNS[:,:,:8], atol=1e-6)
    np.testing.assert_allclose(Trc, T[:,::-1]*SIGNS[:,:,8:], atol=1e-6)
    assert B.shape[1:] == (143,8) and T.shape[1:] == (144,6)
    assert all(np.isfinite(v).all() for v in [B,T,Brc,Trc])
    
    # EP remains in the source-file audit above but is excluded from geometry inputs.
    G=B[:,:,[BP.index(n) for n in GEOMETRY_BP]]
    Grc=Brc[:,:,[BP.index(n) for n in GEOMETRY_BP]]
    E=B[:,:,BP.index("EP"):BP.index("EP")+1]
    Erc=Brc[:,:,BP.index("EP"):BP.index("EP")+1]
    assert G.shape[1:]==(143,7) and E.shape[1:]==(143,1)
    arrays=dict(sequence=(X,Xrc),kmers=(K,Krc),bp=(G,Grc),step=(T,Trc),ep=(E,Erc))
    return dict(arrays=arrays,y=y,ids=np.asarray(ids),canon=np.asarray(canon),
                groups=GROUP_IDS,manifest=manifest,conflicting_groups=conflicts,excluded=excluded,source_indices=source_indices)


class Dataset:
    """Native arrays and train-only forward/RC normalization."""
    def __init__(self, folder, invalid_policy="exclude"):
        self.path=Path(folder).expanduser().resolve()
        raw=load_species(self.path,invalid_policy)
        if raw['conflicting_groups'].size:
            raise ValueError('Identical/RC sequences have conflicting labels; resolve the dataset first.')
        self.__dict__.update(raw)
        self.dataset_id=dataset_identity(self.path)

    def fit_scalers(self, indices, keys):
        result={}
        for key in keys:
            if key=='sequence':continue
            values=np.concatenate([v[indices] for v in self.arrays[key]])
            axes=tuple(range(values.ndim-1))
            result[key+'_mean']=values.mean(axis=axes).astype(np.float32)
            result[key+'_std']=np.maximum(values.std(axis=axes),1e-6).astype(np.float32)
        return result

    def views(self, indices, scalers, keys):
        views=[]
        for strand in (0,1):
            view={}
            for key in keys:
                x=self.arrays[key][strand][indices]
                if key!='sequence':x=(x-scalers[key+'_mean'])/scalers[key+'_std']
                view[key]=x.astype(np.float32)
            views.append(view)
        return tuple(views)


def dataset_identity(path):
    parts=Path(path).resolve().parts
    for i,p in enumerate(parts):
        if p in ('DatasetNup_1','DatasetNup_2','DatasetNup_3'):
            return '/'.join(parts[i:]) if i+1<len(parts) else p+'/Homo_sapiens_hg38'
    return 'Custom/'+Path(path).resolve().name


def catalog(data_root):
    return pd.DataFrame([dict(dataset_id=dataset_identity(p.parent),dataset_path=str(p.parent.resolve()),
                             fasta=next(p.glob('*.fas')).name)
        for p in sorted(Path(data_root).glob('**/sequence')) if list(p.glob('*.fas'))])


def make_splits(data, k=5, seed=42):
    """Stratified outer CV; each outer training set reserves 10% tune, 5% calibration.

    Canonical sequence/RC groups stay intact in every role. Fractions are exact
    (up to rounding) for singleton groups, approximate when groups have repeats.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    if not isinstance(k,int) or isinstance(k,bool) or k<2:raise ValueError('K_FOLDS must be an integer >=2.')
    groups,first=np.unique(data.groups,return_index=True)
    gy=data.y[first]
    if np.bincount(gy,minlength=2).min()<k:raise ValueError('Not enough distinct groups per class for this k.')
    repeated=len(groups)<len(data.y)
    # Stratify unique, homogeneous canonical groups. This scales to the large
    # DatasetNup_2 units without iterative group assignment for every inner fold.
    units=groups if repeated else np.arange(len(data.y))
    labels=gy if repeated else data.y
    splitter=StratifiedKFold(k,shuffle=True,random_state=seed)
    result=[]
    for fold,(outer_pos,test_pos) in enumerate(splitter.split(units,labels),1):
        outer,test=units[outer_pos],units[test_pos]
        outer_labels=labels[outer_pos]
        train,held=train_test_split(outer,test_size=.15,stratify=outer_labels,random_state=seed)
        held_labels=gy[np.searchsorted(groups,held)] if repeated else data.y[held]
        tune,cal=train_test_split(held,test_size=1/3,stratify=held_labels,random_state=61000+fold)
        if repeated:
            train,tune,cal,test=[np.flatnonzero(np.isin(data.groups,u)) for u in (train,tune,cal,test)]
        record=dict(fold=fold,train=list(map(int,train)),tune=list(map(int,tune)),
                    calibration=list(map(int,cal)),test=list(map(int,test)))
        result.append(record)
    validate_splits(data,result)
    return result


def validate_splits(data, parts):
    import itertools
    all_indices=set(range(len(data.y)))
    for p in parts:
        roles=[np.asarray(p[r],int) for r in ('train','tune','calibration','test')]
        if set.union(*(set(r) for r in roles))!=all_indices or sum(map(len,roles))!=len(data.y):
            raise ValueError('Split rows overlap or are missing.')
        for a,b in itertools.combinations(roles,2):
            if set(data.groups[a])&set(data.groups[b]):raise ValueError('Canonical group leakage.')
        if any(len(np.unique(data.y[r]))!=2 for r in roles):raise ValueError('Every role must contain both classes.')
    if sorted(i for p in parts for i in p['test'])!=list(range(len(data.y))):
        raise ValueError('Outer test rows must cover each sample once.')
