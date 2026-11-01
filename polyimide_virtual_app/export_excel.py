#!/usr/bin/env python3
"""Export training data, measurement audits, virtual polymers and monomers to one workbook.

 .venv/bin/python polyimide_virtual_app/export_excel.py
"""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
MODELS=BASE/'polyimide_models';SOURCE=BASE/'polyimide_tg_modulus';MONOMERS=BASE/'tci_polyimide/output'
OUTPUT=ROOT/'polyimide_training_and_virtual_data.xlsx'
FONT='Arial'
MAX_WIDTH=60
RULE_LABELS={'non_tensile_method':'引張試験でない測定','crosslinked':'架橋体','secant_inconsistent':'応力とひずみの矛盾',
             'strength_implausible':'単位異常の疑い','low_modulus_review':'要確認文献の低弾性値'}

def read_tsv(path):
    return pd.read_csv(path,sep='\t',encoding='utf-8-sig',keep_default_na=False,low_memory=False)

def text_list(value):
    """Show a JSON list as readable text; SMILES and fractions keep their order."""
    return '; '.join(str(x) for x in json.loads(value)) if value else ''

def training_sheet(target,unit):
    d=read_tsv(MODELS/f'{target}_dataset.tsv')
    return pd.DataFrame({'行ID':d.row_id,'ポリマー種別':d.polymer_kind,'反復単位SMILES':d.components.map(text_list),'モル分率':d.mole_fractions.map(text_list),
        '共重合様式':d.sequence_type,f'学習に使う値_中央値 ({unit})':pd.to_numeric(d.value),f'最小 ({unit})':pd.to_numeric(d.value_min),f'最大 ({unit})':pd.to_numeric(d.value_max),
        '測定数':pd.to_numeric(d.measurement_count),'文献No.':d.reference_nos.map(text_list),'形状':d.shapes.map(text_list),'構造ファミリー':d.family_group})

def main():
    m=read_tsv(SOURCE/'polyimide_measurements.tsv').set_index('measurement_id')
    s=read_tsv(SOURCE/'polyimide_samples.tsv').set_index('sample_id')
    measurement_columns={'sample_id':'試料ID','reference_no':'文献No.','reference_polymer_name':'文献記載のポリマー名','polymer_kind':'ポリマー種別','smiles':'SMILES',
                         'temperature_min_raw':'測定温度 (℃)','measurement_method':'測定法','measurement_standard':'測定規格'}
    # Tg film: every measurement behind the training rows.
    tg=read_tsv(MODELS/'Tg_film_dataset.tsv')
    tg_ids=[i for r in tg.measurement_ids for i in json.loads(r)]
    tg_meas=m.loc[tg_ids,list(measurement_columns)+['value_min','qualifier']].reset_index()
    tg_meas=tg_meas.rename(columns={**measurement_columns,'measurement_id':'測定ID','value_min':'Tg (℃)','qualifier':'付記'})
    tg_meas['Tg (℃)']=pd.to_numeric(tg_meas['Tg (℃)'])
    tg_meas['形状']=tg_meas['試料ID'].map(s.sample_shape_raw)
    # Modulus film: audited measurements plus those outside the 0.01-10 GPa range.
    audit=read_tsv(MODELS/'tensile_modulus_film_audit.tsv')
    flags=[(audit[k].astype(str)=='True') for k in RULE_LABELS]
    audit['reason']=['; '.join(label for label,flag in zip(RULE_LABELS.values(),row) if flag) for row in zip(*flags)]
    removed=read_tsv(MODELS/'tensile_modulus_film_range_removed_measurements.tsv')
    extra=pd.DataFrame({'measurement_id':removed.measurement_id,'modulus':pd.to_numeric(removed.value_GPa),'reason':'10 GPa以上'})
    extra['sample_id']=extra.measurement_id.map(m.sample_id);extra['reference_no']=extra.measurement_id.map(m.reference_no);extra['measurement_method']=extra.measurement_id.map(m.measurement_method)
    all_modulus=pd.concat([audit,extra],ignore_index=True)
    modulus=pd.DataFrame({'測定ID':all_modulus.measurement_id,'試料ID':all_modulus.sample_id,'文献No.':all_modulus.reference_no,
        '文献記載のポリマー名':all_modulus.measurement_id.map(m.reference_polymer_name),'SMILES':all_modulus.measurement_id.map(m.smiles),
        '弾性率 (GPa)':pd.to_numeric(all_modulus.modulus),'強度 (GPa)':pd.to_numeric(all_modulus.strength,errors='coerce'),'破断伸び (%)':pd.to_numeric(all_modulus.elongation,errors='coerce'),
        '強度÷伸び÷弾性率':pd.to_numeric(all_modulus.secant_ratio,errors='coerce'),'測定法':all_modulus.measurement_method,
        '測定温度 (℃)':all_modulus.measurement_id.map(m.temperature_min_raw),'形状':all_modulus.sample_id.map(s.sample_shape_raw),
        'モデルで使用':all_modulus.reason.map(lambda r:'除外' if r else '使用'),'除外理由':all_modulus.reason})
    # CTE: all unique source records with the reason for each decision.
    c=read_tsv(MODELS/'CTE_film_audit.tsv')
    reasons={'included':'','not_film':'フィルム以外','stated_range_excludes_room_temperature':'室温を含まない温度範囲','incomplete_smiles':'SMILESが不完全','component_alignment':'組成と成分が対応しない',
             'above_glassy_limit':'150 ppm/℃超','not_linear':'架橋または超分岐','not_thermomechanical':'熱機械測定でない','censored_value':'不等号付きの値','unit_unknown':'単位不明','range_or_missing_value':'範囲値または欠損'}
    assert set(c.reason)<=set(reasons)
    cte=pd.DataFrame({'測定ID':c.property_id,'試料ID':c.sample_id,'文献No.':c.reference_no,'文献記載のポリマー名':c.polymer,'SMILES':c.sample_id.map(s.smiles),
        'CTE (ppm/℃)':pd.to_numeric(c.CTE_ppm_per_C),'測定法':c.method,'記載の温度範囲 (℃)':c.stated_range,'形状':c['shape'],
        'モデルで使用':c.reason.map(lambda r:'使用' if r=='included' else '除外'),'除外理由':c.reason.map(reasons)})
    # Virtual polymers with both monomer SMILES.
    v=read_tsv(ROOT/'virtual_predictions.tsv')
    number=lambda col:pd.to_numeric(v[col])
    virtual=pd.DataFrame({'酸二無水物_製品番号':v.dianhydride_id,'酸二無水物名':v.dianhydride_name,'酸二無水物SMILES':v.dianhydride_smiles,
        'ジアミン_製品番号':v.diamine_id,'ジアミン名':v.diamine_name,'ジアミンSMILES':v.diamine_smiles,'ポリイミドp-SMILES':v.psmiles,'別の位置異性体p-SMILES':v.alt_psmiles,
        '繰り返し単位分子量':number('repeat_unit_mw'),'Tg予測 (℃)':number('Tg_film_pred'),'弾性率予測 (GPa)':number('tensile_modulus_film_clean_pred'),'CTE予測 (ppm/℃)':number('CTE_film_pred'),
        'Tg類似度':number('Tg_film_max_similarity'),'弾性率類似度':number('tensile_modulus_film_clean_max_similarity'),'CTE類似度':number('CTE_film_max_similarity')})
    def monomer_sheet(name):
        d=pd.read_csv(MONOMERS/name,encoding='utf-8-sig',keep_default_na=False)
        return pd.DataFrame({'製品番号':d.tci_product_numbers,'名称':d['name'],'CAS':d.cas_rn,'SMILES':d.smiles,'塩を除いたSMILES':d.parent_smiles,'形態':d.form,
            'ポリイミド生成の判定':d.polyimide_ok.map(lambda x:'可' if str(x)=='True' else '不可'),'注意フラグ':d['flags'],'種類':d.type_detail,'分子式':d.formula,'分子量':pd.to_numeric(d.mw),
            'CID':d.cid,'PubChem登録日':d.pubchem_deposit_date,'製品ページURL':d.product_url,'IUPAC名':d.iupac_name})
    sheets={'学習データ_Tg':training_sheet('Tg_film','℃'),'学習データ_弾性率':training_sheet('tensile_modulus_film_clean','GPa'),'学習データ_CTE':training_sheet('CTE_film','ppm/℃'),
            '測定データ_Tg':tg_meas,'測定データ_弾性率':modulus,'測定データ_CTE':cte,'仮想ポリマー':virtual,'酸二無水物':monomer_sheet('tci_dianhydrides.csv'),'ジアミン':monomer_sheet('tci_diamines.csv')}
    descriptions={'学習データ_Tg':'Tgモデルの学習行。同じ構造・組成の測定をまとめた中央値','学習データ_弾性率':'弾性率モデルの学習行(クレンジング後)','学習データ_CTE':'CTEモデルの学習行(クレンジング後)',
        '測定データ_Tg':'Tg学習行のもとになった測定','測定データ_弾性率':'弾性率の測定と採否。0.01 GPa以下は以前の工程で除外済みのため含まない','測定データ_CTE':'CTEの測定(重複除去後)と採否',
        '仮想ポリマー':'TCI登録モノマーの全組み合わせ。予測はフィルムモデル','酸二無水物':'PubChemのTCI登録から抽出した酸二無水物(全件)','ジアミン':'PubChemのTCI登録から抽出したジアミン(全件)'}
    with pd.ExcelWriter(OUTPUT,engine='xlsxwriter',engine_kwargs={'options':{'strings_to_formulas':False,'strings_to_urls':False}}) as writer:
        book=writer.book
        base=book.formats[0];base.set_font_name(FONT);base.set_font_size(10)
        header=book.add_format({'bold':True,'font_name':FONT,'font_size':10,'bg_color':'#E4E3DF','bottom':1,'text_wrap':True,'valign':'top'})
        cell=book.add_format({'font_name':FONT,'font_size':10})
        title=book.add_format({'bold':True,'font_name':FONT,'font_size':14})
        bold=book.add_format({'bold':True,'font_name':FONT,'font_size':10})
        wrap=book.add_format({'font_name':FONT,'font_size':10,'text_wrap':True,'valign':'top'})
        guide=book.add_worksheet('説明');writer.sheets['説明']=guide
        guide.set_column(0,0,22,cell);guide.set_column(1,1,10,cell);guide.set_column(2,2,90,cell)
        guide.write(0,0,'ポリイミド 学習データと仮想ポリマー',title)
        guide.write_row(2,0,['シート','行数','内容'],header)
        for i,name in enumerate(sheets):
            guide.write(3+i,0,name,cell)
            guide.write_formula(3+i,1,f"=COUNTA('{name}'!A:A)-1",cell,len(sheets[name]))
            guide.write(3+i,2,descriptions[name],cell)
        notes=['元データ: data_20240322.tsv(polyimide_tg_modulus で無添加ポリイミドを抽出したもの)',
               '弾性率の除外: 0.01 GPa以下または10 GPa以上、引張試験でない測定、架橋体、応力とひずみの矛盾、単位異常の疑い、要確認文献の低弾性値',
               'CTE: 元データの大半に測定温度の記載がなく、厳密な室温の値ではない(通常のTMAの報告値)',
               '類似度: 学習データ内で最も近い反復単位とのMorgan Tanimoto類似度。低いほど外挿',
               '販売状況: TCIがPubChemに登録した情報に基づく。現在の販売は製品ページURLで確認が必要',
               '予測値: 構造のみに基づく参考値。分子量、配向、イミド化条件は考慮していない',
               '弾性率モデルの実測中央値は、同じ構造でも測定条件で大きく異なる場合がある']
        guide.write(4+len(sheets),0,'注意',bold)
        for i,note in enumerate(notes):guide.merge_range(5+len(sheets)+i,0,5+len(sheets)+i,2,note,wrap)
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name]
            for j,col in enumerate(frame.columns):
                sheet.write(0,j,col,header)
                width=max(len(str(col))*1.6,frame[col].astype(str).str.len().quantile(.9)+2)
                sheet.set_column(j,j,min(MAX_WIDTH,width),cell)
            sheet.set_row(0,32);sheet.freeze_panes(1,0);sheet.autofilter(0,0,len(frame),len(frame.columns)-1)
    print({k:len(v) for k,v in sheets.items()},round(OUTPUT.stat().st_size/1e6,1),'MB')

if __name__=='__main__':main()
