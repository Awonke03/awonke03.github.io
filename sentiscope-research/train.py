"""Train portable text classifiers; source data never committed."""
import csv, hashlib, io, json, re, unicodedata, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'sentiscope'
D=8192
SEED=42
SOURCES={}
def download(url):
    with urllib.request.urlopen(url,timeout=120) as r: raw=r.read()
    SOURCES[url]=hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')
def norm(t): return unicodedata.normalize('NFKC',t).lower()
def tokens(t): return re.findall(r"[^\W_]+",norm(t),flags=re.UNICODE)
def bucket(s):
    h=2166136261
    for c in s: h=((h^ord(c))*16777619)&0xffffffff
    return h%D
def vec(t):
    t=norm(t)[:4000];w=tokens(t)
    f=['w:'+x for x in w]+['b:'+a+' '+b for a,b in zip(w,w[1:])]
    for word in w:
        s='^'+word+'$'
        f.extend('c:'+s[i:i+n] for n in (3,4,5) for i in range(len(s)-n+1))
    c=Counter(bucket(x) for x in f); v={i:np.sqrt(n) for i,n in c.items()}
    n=np.sqrt(sum(x*x for x in v.values())) or 1
    return {i:x/n for i,x in v.items()}
def matrix(texts):
    ix=[]; vals=[]; ptr=[0]
    for t in texts:
        d=vec(t);ix.extend(d);vals.extend(d.values());ptr.append(len(ix))
    return csr_matrix((np.array(vals,dtype=np.float32),ix,ptr),shape=(len(texts),D))
def unique_splits(splits):
    # Hold-out priority: remove duplicate text from earlier partitions.
    seen=set(); out=[]
    for rows in reversed(splits):
        new=[]
        for t,y in rows:
            k=' '.join(tokens(t))
            if k and k not in seen: new.append((t,y));seen.add(k)
        out.append(new)
    return list(reversed(out))
def tweet():
    base='https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/'
    return [[(t,['negative','neutral','positive'][int(y)]) for t,y in zip(download(base+s+'_text.txt').splitlines(),download(base+s+'_labels.txt').splitlines(),strict=True)] for s in ['train','val','test']]
def tsonga():
    base='https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/tso/'
    return [[(r['tweet'],r['label'].strip().lower()) for r in csv.DictReader(io.StringIO(download(base+s+'.tsv')),delimiter='\t') if r.get('tweet') and r.get('label','').strip().lower() in ['negative','neutral','positive']] for s in ['train','dev','test']]
def portable(m):
    if isinstance(m,MultinomialNB): w=m.feature_log_prob_; b=m.class_log_prior_
    else:w=m.coef_;b=m.intercept_
    return {'classes':m.classes_.tolist(),'w':np.round(w,5).tolist(),'b':np.round(b,5).tolist()}
def sentiment(rows):
    a,v,t=unique_splits(rows); X=matrix([x for x,_ in a]);V=matrix([x for x,_ in v]);T=matrix([x for x,_ in t])
    ya=[y for _,y in a];yv=[y for _,y in v];yt=[y for _,y in t]
    assert len(set(ya))==3 and len(set(yt))==3
    candidates={**{f'logistic_{c}':LogisticRegression(C=c,max_iter=300,class_weight='balanced',random_state=SEED) for c in (.5,2,8)},**{f'nb_{a}':MultinomialNB(alpha=a) for a in (.1,1)},'sgd':SGDClassifier(loss='log_loss',alpha=1e-5,class_weight='balanced',random_state=SEED,max_iter=100)}
    scores={}
    for n,m in candidates.items():
        m.fit(X,ya);scores[n]=float(f1_score(yv,m.predict(V),average='macro'))
    name=max(scores,key=scores.get);m=candidates[name]
    test=classification_report(yt,m.predict(T),output_dict=True,zero_division=0)
    return portable(m),{'winner':name,'validation_macro_f1':scores,'test':test,'counts':[len(a),len(v),len(t)]}
def emotions():
    base='https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/'
    names=download(base+'emotions.txt').splitlines()
    rows=[]
    for s in ['train','dev','test']:
        r=[]
        for t,labels,_ in csv.reader(io.StringIO(download(base+s+'.tsv')),delimiter='\t',quoting=csv.QUOTE_NONE):
            r.append((t,[int(i) for i in labels.split(',')]))
        rows.append(r)
    a,v,t=unique_splits(rows)
    def xy(r):
        y=np.zeros((len(r),len(names)),dtype=int)
        for i,(_,ids) in enumerate(r): y[i,ids]=1
        return matrix([s for s,_ in r]),y
    X,Y=xy(a);V,U=xy(v);T,Z=xy(t)
    best=None;bestf=-1;scores={}
    for alpha in (1e-4,1e-5):
        m=OneVsRestClassifier(SGDClassifier(loss='log_loss',alpha=alpha,max_iter=80,class_weight='balanced',random_state=SEED),n_jobs=2).fit(X,Y)
        p=m.predict_proba(V)
        th=np.array([max(np.arange(.1,.91,.05),key=lambda q:f1_score(U[:,i],p[:,i]>=q,zero_division=0)) for i in range(len(names))])
        f=f1_score(U,p>=th,average='macro',zero_division=0);scores[str(alpha)]=float(f)
        if f>bestf:best=(m,th);bestf=f
    m,th=best;p=m.predict_proba(T)
    return {'classes':names,'w':np.round(np.vstack([e.coef_[0] for e in m.estimators_]),5).tolist(),'b':np.round([e.intercept_[0] for e in m.estimators_],5).tolist(),'thresholds':th.tolist()},{'validation_macro_f1':scores,'test_macro_f1':float(f1_score(Z,p>=th,average='macro',zero_division=0)),'test_micro_f1':float(f1_score(Z,p>=th,average='micro',zero_division=0)),'counts':[len(a),len(v),len(t)]}
def main():
    OUT.mkdir(exist_ok=True)
    en,er=sentiment(tweet());ts,tr=sentiment(tsonga());em,gr=emotions()
    models={'version':1,'dimension':D,'sentiment':{'en':en,'ts':ts},'emotion':em}
    report={'english':er,'xitsonga_mozambique':tr,'goemotions':gr,'source_sha256':SOURCES,'limitations':'Social media domain, Xitsonga corpus from Mozambique; not South African customer-service validation. No calibrated probability claims. Emotion English only. No raw texts redistributed.'}
    (OUT/'models.json').write_text(json.dumps(models,separators=(',',':')))
    (ROOT/'report.json').write_text(json.dumps(report,indent=2))
    fixtures=['I love the service, thank you!','This is terrible, I want a refund.','The account was opened yesterday.','Ndzа khensa','Great 😃 but expensive']
    (ROOT/'parity.json').write_text(json.dumps([{'text':s,'vector':vec(s)} for s in fixtures]))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
