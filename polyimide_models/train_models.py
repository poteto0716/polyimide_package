#!/usr/bin/env python3
"""Screen feature/model pairs, select on group CV, evaluate a locked holdout."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,collections,gc,hashlib,json,time,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from modeling import FeatureBuilder,FEATURE_SETS,PolymerEnsemble
warnings.filterwarnings('ignore',message='X does not have valid feature names')
warnings.filterwarnings('ignore',message='.*constant.*')
ROOT=Path(__file__).resolve().parent
MODELS=['ridge','elastic_net','svr_rbf','extra_trees','random_forest','hist_gradient_boosting','lightgbm','xgboost','catboost']
SEED=20261031

def make_model(name,variant=0):
    if name=='ridge':reg=Ridge(alpha=10 if not variant else 100)
    elif name=='elastic_net':reg=ElasticNet(alpha=.01 if not variant else .05,l1_ratio=.2,max_iter=12000,tol=1e-4,random_state=SEED)
    elif name=='svr_rbf':reg=SVR(C=10 if not variant else 30,epsilon=.12,kernel='rbf',gamma='scale',cache_size=512)
    elif name=='extra_trees':reg=ExtraTreesRegressor(n_estimators=180,max_features=.8,min_samples_leaf=2 if not variant else 1,n_jobs=2,random_state=SEED)
    elif name=='random_forest':reg=RandomForestRegressor(n_estimators=160,max_features=.7,min_samples_leaf=2 if not variant else 1,n_jobs=2,random_state=SEED)
    elif name=='hist_gradient_boosting':reg=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15 if not variant else 31,l2_regularization=3,min_samples_leaf=15,random_state=SEED)
    elif name=='lightgbm':reg=LGBMRegressor(n_estimators=300,learning_rate=.035,num_leaves=15 if not variant else 31,min_child_samples=12,colsample_bytree=.85,reg_lambda=5,n_jobs=2,verbosity=-1,random_state=SEED)
    elif name=='xgboost':reg=XGBRegressor(n_estimators=300,learning_rate=.035,max_depth=3 if not variant else 4,min_child_weight=4,subsample=.85,colsample_bytree=.8,reg_lambda=5,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=SEED)
    elif name=='catboost':reg=CatBoostRegressor(iterations=260,depth=5 if not variant else 6,learning_rate=.045,l2_leaf_reg=5,loss_function='RMSE',thread_count=2,verbose=False,allow_writing_files=False,random_seed=SEED)
    else:raise ValueError(name)
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('nonconstant',VarianceThreshold())]
    if name in ['ridge','elastic_net','svr_rbf']:steps.append(('scale',StandardScaler()))
    steps.append(('regressor',reg))
    return TransformedTargetRegressor(regressor=Pipeline(steps),transformer=StandardScaler())

def forward(y,transform):return np.log10(y) if transform=='log10' else y

def inverse(y,transform,target):
    y=np.power(10,np.clip(y,-8,8)) if transform=='log10' else y
    return np.maximum(y,0) if target.startswith('tensile_modulus') else y

def metrics(y,p):
    d={'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),
       'median_AE':float(np.median(np.abs(y-p))),'n':len(y)}
    if np.all(y>0) and np.all(p>0):
        d['log10_RMSE']=float(np.sqrt(np.mean((np.log10(y)-np.log10(p))**2)))
        d['median_absolute_percentage_error']=float(np.median(np.abs(y-p)/y)*100)
    return d

def evaluate(candidate,X,y,splits,target):
    start=time.monotonic();oof=np.full(len(y),np.nan);foldscores=[]
    for train,val in splits:
        est=make_model(candidate['model'],candidate.get('variant',0))
        est.fit(X[train],forward(y[train],candidate['target_transform']))
        pred=inverse(est.predict(X[val]),candidate['target_transform'],target)
        oof[val]=pred;foldscores.append(metrics(y[val],pred))
    assert np.isfinite(oof).all()
    result={**candidate,**{'cv_'+k:v for k,v in metrics(y,oof).items()},
            'cv_fold_MAE_std':float(np.std([f['MAE'] for f in foldscores])),
            'seconds':time.monotonic()-start,'fold_metrics_json':json.dumps(foldscores)}
    return result,oof

def bootstrap(y,p,groups,n=500):
    rng=np.random.default_rng(SEED);names=np.unique(groups);by={g:np.flatnonzero(groups==g) for g in names};vals=[]
    for _ in range(n):
        idx=np.concatenate([by[g] for g in rng.choice(names,size=len(names),replace=True)])
        if len(idx)>1:vals.append(metrics(y[idx],p[idx]))
    return {k:[float(v) for v in np.quantile([r[k] for r in vals],[.025,.975])] for k in ['MAE','RMSE','R2']}

def candidate_id(c):return f"{c['feature_set']}__{c['model']}__{c['target_transform']}__v{c.get('variant',0)}"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True,choices=['Tg','Tg_film','tensile_modulus','tensile_modulus_film','tensile_modulus_film_range','tensile_modulus_film_clean','CTE_film']);args=ap.parse_args()
    target=args.target;out=ROOT/target;out.mkdir(exist_ok=True)
    dataset=json.loads((ROOT/'datasets.json').read_text())[target]
    arrays=np.load(ROOT/'features.npz')
    Xs={f:arrays[target+'__'+f] for f in FEATURE_SETS}
    y=np.array([r['value'] for r in dataset]);groups=np.array([r['family_group'] for r in dataset])
    # Split choice balances counts only; no labels or candidate scores are consulted.
    candidates=list(GroupShuffleSplit(n_splits=100,test_size=.2,random_state=SEED).split(y,groups=groups))
    dev,test=min(candidates,key=lambda z:abs(len(z[1])/len(y)-.2))
    assert set(groups[dev]).isdisjoint(set(groups[test]))
    train_units=set(s for i in dev for s in dataset[i]['components']);test_units=set(s for i in test for s in dataset[i]['components'])
    assert train_units.isdisjoint(test_units)
    split3=list(GroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(y[dev],groups=groups[dev]))
    split5=list(GroupKFold(n_splits=5,shuffle=True,random_state=SEED+1).split(y[dev],groups=groups[dev]))
    partition=['development' if i in set(dev) else 'holdout' for i in range(len(y))]
    manifest=pd.DataFrame([{'row_id':r['row_id'],'family_group':r['family_group'],'partition':partition[i],
                            'components_json':json.dumps(r['components']),'mole_fractions_json':json.dumps(r['mole_fractions'])} for i,r in enumerate(dataset)])
    manifest.to_csv(out/'split_manifest.tsv',sep='\t',index=False,encoding='utf-8-sig')
    splitinfo={'seed':SEED,'development_rows':len(dev),'holdout_rows':len(test),'development_families':len(set(groups[dev])),
               'holdout_families':len(set(groups[test])),'shared_repeat_units':len(train_units&test_units),
               'rule':'Connected families of shared canonical repeat-unit SMILES; same repeat unit never crosses split.',
               'selection_metric':'development group-CV MAE in original units','data_rows':len(y)}
    (out/'split_info.json').write_text(json.dumps(splitinfo,indent=2));print('SPLIT',target,json.dumps(splitinfo),flush=True)
    transforms=['identity'] if target.startswith(('Tg','CTE')) else ['identity','log10']
    screen=[]
    for feature in FEATURE_SETS:
        for model in MODELS:
            for transform in transforms:
                c={'feature_set':feature,'model':model,'target_transform':transform,'variant':0}
                ident=candidate_id(c);checkpoint=out/('screen_'+ident+'.json')
                try:
                    if checkpoint.exists():result=json.loads(checkpoint.read_text())
                    else:
                        result,_=evaluate(c,Xs[feature][dev],y[dev],split3,target)
                        checkpoint.write_text(json.dumps(result))
                    screen.append(result)
                    print(f"SCREEN {target} {len(screen)} {ident} MAE={result['cv_MAE']:.5g} sec={result['seconds']:.1f}",flush=True)
                except Exception as exc:
                    print('FAILED',ident,repr(exc),flush=True)
                    with (out/'failures.log').open('a') as f:f.write(ident+' '+repr(exc)+'\n')
                gc.collect()
        pd.DataFrame(screen).sort_values('cv_MAE').to_csv(out/'screening_3fold.tsv',sep='\t',index=False,encoding='utf-8-sig')
    screen_df=pd.DataFrame(screen).sort_values('cv_MAE')
    short=[]
    for feature in FEATURE_SETS:
        for _,r in screen_df[screen_df.feature_set==feature].head(2).iterrows():
            for variant in [0,1]:
                short.append({'feature_set':feature,'model':r['model'],'target_transform':r['target_transform'],'variant':variant})
    refined=[];oofs={}
    for j,c in enumerate(short):
        ident=candidate_id(c);checkpoint=out/('refined_'+ident+'.json');npfile=out/('oof_'+ident+'.npy')
        if checkpoint.exists() and npfile.exists():result=json.loads(checkpoint.read_text());oof=np.load(npfile)
        else:
            result,oof=evaluate(c,Xs[c['feature_set']][dev],y[dev],split5,target)
            checkpoint.write_text(json.dumps(result));np.save(npfile,oof)
        refined.append(result);oofs[ident]=oof
        print(f"REFINE {target} {j+1}/{len(short)} {ident} MAE={result['cv_MAE']:.5g} sec={result['seconds']:.1f}",flush=True)
        pd.DataFrame(refined).sort_values('cv_MAE').to_csv(out/'leaderboard_5fold.tsv',sep='\t',index=False,encoding='utf-8-sig')
    rank=pd.DataFrame(refined).sort_values('cv_MAE')
    perfeature=rank.drop_duplicates('feature_set').copy();top=perfeature.head(5)
    assert len(top)==5
    top.to_csv(out/'top5_feature_model_pairs.tsv',sep='\t',index=False,encoding='utf-8-sig')
    perfeature.to_csv(out/'feature_set_ranking.tsv',sep='\t',index=False,encoding='utf-8-sig')
    selected=top.to_dict('records');best=rank.iloc[0].to_dict();best_id=candidate_id(best)
    oofmat=np.array([oofs[candidate_id(c)] for c in selected]).T
    equal=np.ones(5)/5;inv=1/np.array([c['cv_MAE'] for c in selected]);inv/=inv.sum()
    equal_mae=mean_absolute_error(y[dev],oofmat@equal);inv_mae=mean_absolute_error(y[dev],oofmat@inv)
    weights=inv if inv_mae<equal_mae else equal
    weight_method='inverse_CV_MAE' if inv_mae<equal_mae else 'equal'
    # Freeze member selection and weights before any holdout prediction is made.
    selection={'target':target,'members':[{'candidate_id':candidate_id(c),'feature_set':c['feature_set'],'model':c['model'],
                'target_transform':c['target_transform'],'variant':c['variant'],'cv_MAE':c['cv_MAE'],'weight':float(w)} for c,w in zip(selected,weights)],
               'best_single_candidate':best_id,'weight_method':weight_method,
               'ensemble_selection_cv':metrics(y[dev],oofmat@weights),
               'equal_weight_selection_cv_MAE':float(equal_mae),'inverse_weight_selection_cv_MAE':float(inv_mae),
               'note':'CV scores are used for model/feature selection; holdout is the generalization estimate.'}
    (out/'frozen_selection.json').write_text(json.dumps(selection,indent=2))
    members=[];test_predictions=[];test_rows=[]
    for c in selected:
        est=make_model(c['model'],int(c['variant']));X=Xs[c['feature_set']]
        est.fit(X[dev],forward(y[dev],c['target_transform']))
        pred=inverse(est.predict(X[test]),c['target_transform'],target)
        test_predictions.append(pred)
        members.append({'feature_set':c['feature_set'],'target_transform':c['target_transform'],'estimator':est,'candidate_id':candidate_id(c)})
        test_rows.append({'name':candidate_id(c),**metrics(y[test],pred),'cv_MAE':c['cv_MAE']})
    testmat=np.array(test_predictions).T;ensemble_pred=testmat@weights
    best_index=[candidate_id(c) for c in selected].index(best_id)
    best_pred=testmat[:,best_index]
    baseline=DummyRegressor(strategy='median').fit(np.ones((len(dev),1)),y[dev]).predict(np.ones((len(test),1)))
    summary={'target':target,'unit':'C' if target.startswith('Tg') else 'ppm_per_C' if target.startswith('CTE') else 'GPa','split':splitinfo,'selection':selection,
             'screening_combinations':len(screen),'refined_combinations':len(refined),
             'baseline_holdout':metrics(y[test],baseline),'best_single_holdout':metrics(y[test],best_pred),
             'ensemble_holdout':metrics(y[test],ensemble_pred),
             'ensemble_holdout_95pct_group_bootstrap':bootstrap(y[test],ensemble_pred,groups[test]),
             'holdout_members':test_rows}
    for label,pred in [('best_single',best_pred),('ensemble',ensemble_pred)]:
        for kind in ['homopolymer','copolymer']:
            mask=np.array([dataset[i]['polymer_kind']==kind for i in test])
            if mask.sum()>1:summary[label+'_holdout_'+kind]=metrics(y[test][mask],pred[mask])
        if target=='tensile_modulus':
            mask=np.array([all(sh in ['Film','Sheet','Membrane','membrane','Foil'] for sh in dataset[i]['shapes']) for i in test])
            if mask.sum()>1:summary[label+'_holdout_film_membrane']=metrics(y[test][mask],pred[mask])
    eval_ensemble=PolymerEnsemble(members,weights,target,metadata={'fit_partition':'development','selection':selection})
    joblib.dump(eval_ensemble,out/'ensemble_evaluation.joblib',compress=3)
    best_evaluator=PolymerEnsemble([members[best_index]],[1],target,metadata={'fit_partition':'development'})
    joblib.dump(best_evaluator,out/'best_single_evaluation.joblib',compress=3)
    records=[{'components':dataset[i]['components'],'mole_fractions':dataset[i]['mole_fractions'],'sequence_type':dataset[i]['sequence_type']} for i in test]
    reloaded=joblib.load(out/'ensemble_evaluation.joblib')
    np.testing.assert_allclose(reloaded.predict(records),ensemble_pred,rtol=2e-5,atol=1e-6)
    predrows=[]
    for j,i in enumerate(test):
        r=dataset[i]
        predrows.append({'row_id':r['row_id'],'family_group':r['family_group'],'polymer_kind':r['polymer_kind'],
                         'components_json':json.dumps(r['components']),'mole_fractions_json':json.dumps(r['mole_fractions']),
                         'sequence_type':r['sequence_type'],'observed':y[i],'best_single_prediction':best_pred[j],
                         'ensemble_prediction':ensemble_pred[j],'ensemble_error':ensemble_pred[j]-y[i],
                         'shapes_json':json.dumps(r['shapes']),'reference_nos_json':json.dumps(r['reference_nos']),
                         **{f'member_{k+1}_prediction':testmat[j,k] for k in range(5)}})
    pd.DataFrame(predrows).to_csv(out/'holdout_predictions.tsv',sep='\t',index=False,encoding='utf-8-sig')
    pd.DataFrame({'row_id':[dataset[i]['row_id'] for i in dev],'observed':y[dev],
                  'best_single_oof':oofs[best_id],'ensemble_oof':oofmat@weights,
                  **{f'member_{k+1}_oof':oofmat[:,k] for k in range(5)}}).to_csv(out/'selection_oof_predictions.tsv',sep='\t',index=False,encoding='utf-8-sig')
    # An interpretable, compact feature set provides individual-feature rankings.
    interpret=rank[rank.feature_set=='chemistry'].iloc[0].to_dict()
    est=make_model(interpret['model'],int(interpret['variant']));X=Xs['chemistry']
    est.fit(X[dev],forward(y[dev],interpret['target_transform']))
    def scorer(e,X0,y0):return -mean_absolute_error(y0,inverse(e.predict(X0),interpret['target_transform'],target))
    importance=permutation_importance(est,X[test],y[test],scoring=scorer,n_repeats=5,random_state=SEED,n_jobs=2)
    feature_names=FeatureBuilder().names('chemistry')
    fi=pd.DataFrame({'feature':feature_names,'holdout_MAE_increase':importance.importances_mean,'permutation_std':importance.importances_std}).sort_values('holdout_MAE_increase',ascending=False)
    fi.to_csv(out/'interpretable_feature_importance.tsv',sep='\t',index=False,encoding='utf-8-sig')
    summary['interpretable_importance_model']=candidate_id(interpret)
    summary['individual_features_top5']=fi.head(5).to_dict('records')
    # Production artifacts use all eligible labels only after holdout assessment.
    production=[]
    for c in selected:
        est=make_model(c['model'],int(c['variant']));est.fit(Xs[c['feature_set']],forward(y,c['target_transform']))
        production.append({'feature_set':c['feature_set'],'target_transform':c['target_transform'],'estimator':est,'candidate_id':candidate_id(c)})
    joblib.dump(PolymerEnsemble(production,weights,target,metadata={'fit_partition':'all_eligible_data','evaluation_artifact':'ensemble_evaluation.joblib','selection':selection}),out/'ensemble_production.joblib',compress=3)
    joblib.dump(PolymerEnsemble([production[best_index]],[1],target,metadata={'fit_partition':'all_eligible_data'}),out/'best_single_production.joblib',compress=3)
    example_records=[{'components':r['components'],'mole_fractions':r['mole_fractions'],'sequence_type':r['sequence_type']} for r in dataset[:3]]
    (out/'example_input.json').write_text(json.dumps(example_records,indent=2))
    summary['artifact_reload_prediction_check']='passed'
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('RESULT',target,json.dumps({k:summary[k] for k in ['baseline_holdout','best_single_holdout','ensemble_holdout']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
