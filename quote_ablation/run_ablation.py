from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from compare_models import Config, Features, FOLDS, make_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results'));a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'protocol.json').exists(): raise SystemExit('Choose a new output directory')
 cfg=Config('cat','raw','abs_median')
 protocol={'folds':FOLDS,'removed':['quote_signal','quote_based_rate'],'other_features':'unchanged v2 selected pipeline','max_iterations':800,'patience':60,'seed':42,'threads':4,'input_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'Post-v2 development ablation. No October evaluation or final prediction changes. Both variants rerun under same conditions.'}
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 df=pd.read_csv(a.data,parse_dates=['date']);df=df[df.date<'2025-10-01'].sort_values(['date','load_id'])
 rows=[]
 for variant in ['with_quote','without_quote']:
  def transform(prep,data):
   x=prep.transform(data)
   return x.drop(columns=['quote_signal','quote_based_rate']) if variant=='without_quote' else x
  for fold in FOLDS:
   print(variant,fold['name'],flush=True)
   train=df[df.date<fold['train_end']];inner=train[train.date<fold['inner_cut']];stop=train[train.date>=fold['inner_cut']]
   prep=Features(cfg).fit(inner);model=make_model(cfg,800,4)
   model.fit(transform(prep,inner),inner.posted_rate,eval_set=(transform(prep,stop),stop.posted_rate),early_stopping_rounds=60)
   n=model.get_best_iteration()+1
   prep=Features(cfg).fit(train);model=make_model(cfg,n,4);model.fit(transform(prep,train),train.posted_rate)
   outer=df[(df.date>=fold['train_end'])&(df.date<fold['eval_end'])]
   out=outer[['load_id','date','posted_rate']].copy();out['predicted_rate']=model.predict(transform(prep,outer))
   assert np.isfinite(out.predicted_rate).all()
   out.to_csv(a.output/f'{variant}_{fold["name"]}.csv',index=False)
   for month,g in out.groupby(out.date.dt.strftime('%Y-%m')):
    err=g.predicted_rate-g.posted_rate
    row={'variant':variant,'month':month,'n':len(g),'rounds':n,'mae':abs(err).mean(),'rmse':np.sqrt((err**2).mean())};rows.append(row);print(row,flush=True)
 pd.DataFrame(rows).to_csv(a.output/'metrics.csv',index=False)
 print(pd.DataFrame(rows).groupby('variant')[['mae','rmse']].mean().to_string(),flush=True)
if __name__=='__main__':main()
