#!/usr/bin/env python3
"""Extract traceable polyimide repeat-unit SMILES, Tg and moduli from local files.

Run from the directory containing the two inputs:
    .venv/bin/python polyimide_tg_modulus/extract_polyimide.py
No literature values or structures from external sources are added.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import openpyxl
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')
CU_RE = re.compile(r'[CJ]U\d{6}')
PID_RE = re.compile(r'P\d{6}')
IMIDE = Chem.MolFromSmarts('[#7;R]([#6;R](=O))[#6;R](=O)')
PLACEHOLDER_RE = re.compile(r'\bL1[a-z]?\b', re.I)


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    return format(value, '.10g')


def split_top(text, delimiter='/'):
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def unit_kind(raw):
    parts = [x.strip().lower() for x in raw.split('/')]
    if parts and all(x in ('mol%', 'mol ratio', 'mol', 'mole fraction', 'repeat unit ratio') for x in parts):
        return 'mol'
    if parts and all(x in ('wt%', 'wt ratio') for x in parts):
        return 'weight'
    return 'unknown'


def composition_values(raw):
    # Parentheses denote grouped components; they must not be silently flattened.
    parts = split_top(raw) if raw else []
    vals = [number(p) for p in parts]
    if vals and all(x is not None and x >= 0 for x in vals) and sum(vals) > 0:
        return vals
    return None


def read_units(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    units, pid_map = {}, collections.defaultdict(list)
    simple_sheet = {r[0]: r[1] for r in list(book['Sheet1'].values)[1:]}
    for line, (cu, pid, raw, alt) in enumerate(list(book['Sheet2'].values)[1:], 2):
        text = raw or ''
        origin = 'Sheet2!C' + str(line)
        if not text or text.startswith('Unable'):
            text = alt or ''
            origin = 'Sheet2!D' + str(line) if alt else origin
        mol = Chem.MolFromSmiles(text) if text else None
        matches = mol.GetSubstructMatches(IMIDE) if mol else []
        dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0] if mol else []
        backbone = False
        if matches:
            fragments = Chem.GetMolFrags(mol)
            for i, a in enumerate(dummies):
                for b in dummies[i + 1:]:
                    if any(a in f and b in f for f in fragments):
                        path_atoms = set(Chem.GetShortestPath(mol, a, b))
                        if any(match[0] in path_atoms for match in matches):
                            backbone = True
        # Zero-mass dummy atoms represent the polymer connections, not H caps.
        mw_ok = bool(mol and len(dummies) == 2 and len(Chem.GetMolFrags(mol)) == 1
                     and all(mol.GetAtomWithIdx(i).GetDegree() == 1 for i in dummies)
                     and all(a.GetIsotope() == 0 for a in mol.GetAtoms()))
        units[cu] = {
            'component_id': cu, 'registry_pid': pid, 'smiles_raw': raw or '',
            'smiles_selected': text, 'canonical_smiles': Chem.MolToSmiles(mol) if mol else '',
            'smiles_valid': bool(mol), 'cyclic_imide': bool(matches),
            'backbone_imide': backbone, 'dummy_atom_count': len(dummies) if mol else '',
            'repeat_unit_mw_g_mol': Descriptors.MolWt(mol) if mw_ok else '',
            'excel_cell': origin, 'sheet1_agrees': simple_sheet.get(cu) == raw,
        }
        pid_map[pid].append(cu)
    book.close()
    return units, pid_map


def extract_ids(text, pid_map):
    """Preserve textual component order, including repeated units."""
    ids, unknown = [], []
    for token in re.findall(r'[CJ]U\d{6}|P\d{6}|\bL1[a-z]?\b', text, re.I):
        if CU_RE.fullmatch(token):
            ids.append(token)
        elif PID_RE.fullmatch(token):
            mapped = pid_map.get(token, [])
            if len(mapped) == 1:
                ids.extend(mapped)
            else:
                unknown.append(token)
        else:
            unknown.append(token)
    return ids, unknown


def map_components(row, units, pid_map):
    registry, composition = row[4], row[34]
    blend = '//' in registry or bool(row[7]) or '//' in row[10]
    if composition:
        ids, unknown = extract_ids(composition, pid_map)
        groups = split_top(composition)
        complete = bool(ids) and not unknown and all(x in units for x in ids)
        if re.sub(r'[CJ]U\d{6}|[()/\s]', '', composition):
            complete = False
        return ids, complete, 'composition_id', composition, groups, blend
    if registry in pid_map:
        ids = pid_map[registry]
        return ids, len(ids) == 1, 'registry_pid', registry, [registry], blend
    # Explicit CU lists in the source names are usable even if composition_id is blank.
    for index in (10, 9):
        text = row[index]
        if CU_RE.search(text):
            ids = CU_RE.findall(text)
            explicit_list = re.fullmatch(r'\s*(?:P\d{6})?\(?\s*[CJ]U\d{6}(?:\s*/\s*[CJ]U\d{6})*\s*\)?\s*', text)
            complete = bool(explicit_list) and not PLACEHOLDER_RE.search(text) and all(x in units for x in ids)
            if blend:
                complete = False
            return ids, bool(complete), 'polymer_name' if index == 10 else 'reference_name', text, [], blend
    # A complete slash-separated P-ID expression can also identify a copolymer.
    if row[10]:
        ids, unknown = extract_ids(row[10], pid_map)
        if ids:
            bare = re.fullmatch(r'\s*P\d{6}(\s*/\s*P\d{6})*\s*', row[10])
            complete = bool(bare) and not unknown and not blend
            return ids, complete, 'polymer_name_partial' if not complete else 'polymer_name', row[10], [], blend
    if blend:
        ids, unknown = extract_ids(registry, pid_map)
        return ids, False, 'blend_registry_partial', registry, [], True
    return [], False, 'unresolved', composition, [], blend


# Value index, property-ID index, end of source property block, condition prefix.
PROPERTIES = [
    ('Tg', 232, 231, 237, 'glass_transition_temp'),
    ('tensile_modulus', 646, 645, 684, 'tensile_property'),
    ('fiber_tensile_modulus', 666, 645, 684, 'tensile_property'),
    ('tensile_storage_modulus', 685, 684, 707, 'dynamic_tensile_property'),
    ('tensile_loss_modulus', 689, 684, 707, 'dynamic_tensile_property'),
    ('shear_modulus', 708, 707, 726, 'shear_property'),
    ('shear_storage_modulus', 727, 726, 749, 'dynamic_shear_property'),
    ('shear_loss_modulus', 731, 726, 749, 'dynamic_shear_property'),
    ('flexural_modulus', 750, 749, 768, 'flexural_property'),
    ('flexural_storage_modulus', 769, 768, 791, 'dynamic_flexural_property'),
    ('flexural_loss_modulus', 773, 768, 791, 'dynamic_flexural_property'),
    ('compressive_modulus', 792, 791, 810, 'compressive_property'),
    ('compressive_storage_modulus', 811, 810, 833, 'dynamic_compressive_property'),
    ('compressive_loss_modulus', 815, 810, 833, 'dynamic_compressive_property'),
    ('tensile_creep_modulus', 846, 833, 868, 'tensile_creep'),
    ('bulk_modulus', 1010, 1009, 1024, 'bulk_modulus'),
]
GPA_FACTOR = {'GPa': 1, 'MPa': .001, 'Pa': 1e-9, 'kPa': 1e-6,
              'N/mm2': .001, 'ksi': .006894757293168, 'kpsi': .006894757293168}

COLUMN_GUIDE = {
    'sample_id': '元TSVのSID。成形・処理状態を含む試料ID。',
    'registry_pid': '元TSVのポリマーID。ブレンドでは複数IDを含む場合がある。',
    'polymer_kind': 'homopolymer=単一の有効構成単位、copolymer=共重合体、unknown=判定不能。',
    'sequence_type': 'block/random/statistical/alternating/graft等。併記は元情報の混在を維持。unknown=不明。',
    'sequence_source': '配列分類の根拠。copolymer_category列または名称中の明示的な記載。',
    'smiles': '全有効成分を特定できた場合のSMILES。共重合体は各構成単位をドットで区切った表現。',
    'smiles_complete': '全有効構成単位を特定でき、RDKitでSMILESを解析できた場合True。配列の確定を意味しない。',
    'known_component_smiles': '未同定成分がある試料でも、判明した成分のSMILESを格納。完全構造として使用しない。',
    'component_ids': '全構成単位のID。組成・mole_fractionsの順序に対応。ゼロ組成成分も残す。',
    'active_component_ids': '有効な構成単位ID。数値組成が対応する場合、組成ゼロの成分を除く。',
    'component_mapping_source': '対応根拠。composition_id、registry_pid、polymer_name等。',
    'composition_mol_ratio_reported': '原資料でモル単位が明記された構成単位組成比。原順序でコロン表記。',
    'composition_mol_ratio_calculated': 'CU重量組成を各反復単位の分子量で割り、合計1に正規化した計算値。',
    'composition_mole_fractions_json': 'component_idsと同順の構成単位モル分率。原記載または重量組成から換算。',
    'composition_mol_basis': 'モル比の出所。reported=原記載、calculated=重量組成から換算。alignment_unresolved=成分対応未確定。',
    'feed_mol_ratio_reported': '原料の仕込みモル比。reactant_names_rawの!区切り順。生成ポリマーのCU組成とは別。',
    'composition_id_raw': '元TSVの構成単位ID式。括弧による入れ子を保持。',
    'composition_unit_raw': '元TSVのCU組成単位。mol%、mol ratio、wt%等。',
    'composition_value_raw': '元TSVのCU組成値。曖昧な数値や入れ子を改変せず保持。',
    'Tg_C': 'Tg測定が1件で、上下限が同じ場合のみ表示。複数測定または範囲値はMeasurementsを参照。単位℃。',
    'Tg_qualifier': 'Tgに付くca、≧、≦等の原記載。caは概数。',
    'tensile_modulus_GPa': '引張弾性率が1件で、上下限が同じ場合のみ表示。単位GPa。',
    'tensile_modulus_qualifier': '引張弾性率に付くca、≧、≦等の原記載。',
    'has_modulus': 'いずれかの弾性率/貯蔵弾性率/損失弾性率の測定が存在する。引張Eの有無とは異なる。',
    'material_group': 'neat_polymer=無添加・非ブレンドのNeat resin、その他はmodified_or_composite_or_unspecified。',
    'polyimide_evidence': '主鎖上の環状イミドNをSMILESで確認したか、名称のみかを区別。',
    'property': 'Tg、tensile_modulus、tensile_storage_modulus、flexural_modulus等。測定種類を区別。',
    'property_id': '元TSVの物性測定ID。ID、数値、測定条件がすべて一致する重複を除去。',
    'value_min': '換算後の最小値。Tgは℃、弾性率はGPa。換算不能な単位は空欄。',
    'value_max': '換算後の最大値。範囲は平均せず保持。',
    'qualifier': 'ca、≧、≦等の原記載。数値と併せて解釈する。',
    'unit': '換算後の単位。CまたはGPa。',
    'value_min_raw': '元の測定最小値。', 'value_max_raw': '元の測定最大値。', 'unit_raw': '元の測定単位。',
    'temperature_min_raw': '測定温度の原最小値。Tgそのものではない。',
    'temperature_max_raw': '測定温度の原最大値。', 'temperature_unit_raw': '測定温度の単位。',
    'frequency_min_raw': '動的測定の周波数の原最小値。', 'frequency_max_raw': '動的測定の周波数の原最大値。',
    'frequency_unit_raw': '動的測定の周波数単位。',
    'conditions_json': 'TSVに含まれる測定法、規格、温度、周波数等を英語列名で保持。',
    'source_tsv_rows': '元TSVの論理行番号のJSON配列。日本語ヘッダが1行目、英語ヘッダが2行目。',
    'source_sidps': '測定値を採取した元TSVのSIDPのJSON配列。',
    'source_value_column_1based': '元TSVの最小値列番号（先頭列=1）。',
    'measurements_json': '試料ごとの全Tg・弾性率。完全な長形式はMeasurementsおよびmeasurements.tsvを参照。',
    'repeat_unit_mw_g_mol': '反復単位の式量。結合点*の質量は0。末端水素は付加しない。',
    'mole_fraction': '当該構成単位のモル分率。原記載または重量組成から計算。',
    'excel_cell': '01.smilesList.xlsxで使用したSMILESのセル。',
}


def measurements_from_row(row, en):
    for prop, value_index, id_index, end, prefix in PROPERTIES:
        if not (row[value_index] or row[value_index + 1]):
            continue
        # Tg's fourth field is a standard; its unit is the third field, as for E.
        raw_min, raw_max, qualifier, unit = row[value_index:value_index + 4]
        lower, upper = number(raw_min), number(raw_max)
        normalized_unit, factor, offset = '', None, 0
        if prop == 'Tg':
            if unit in ('C', '°C', 'degC'):
                normalized_unit, factor = 'C', 1
            elif unit == 'K':
                normalized_unit, factor, offset = 'C', 1, -273.15
        elif unit in GPA_FACTOR:
            normalized_unit, factor = 'GPa', GPA_FACTOR[unit]
        conditions = {en[i]: row[i] for i in range(id_index + 1, end)
                      if row[i] and (en[i].startswith(prefix + '_meas_'))}
        yield {
            'property': prop, 'property_id': row[id_index],
            'value_min_raw': raw_min, 'value_max_raw': raw_max,
            'qualifier': qualifier, 'unit_raw': unit,
            'value_min': lower * factor + offset if lower is not None and factor is not None else '',
            'value_max': upper * factor + offset if upper is not None and factor is not None else '',
            'unit': normalized_unit, 'unit_conversion': 'converted' if normalized_unit and normalized_unit != unit else 'as_reported' if normalized_unit else 'unresolved',
            'temperature_min_raw': conditions.get(prefix + '_meas_cond_temp_min', ''),
            'temperature_max_raw': conditions.get(prefix + '_meas_cond_temp_max', ''),
            'temperature_qualifier': conditions.get(prefix + '_meas_cond_temp_inequality_sign', ''),
            'temperature_unit_raw': conditions.get(prefix + '_meas_cond_temp_unit', ''),
            'frequency_min_raw': conditions.get(prefix + '_meas_cond_freq_min', ''),
            'frequency_max_raw': conditions.get(prefix + '_meas_cond_freq_max', ''),
            'frequency_unit_raw': conditions.get(prefix + '_meas_cond_freq_unit', ''),
            'measurement_method': conditions.get(prefix + '_meas_method', ''),
            'measurement_standard': conditions.get(prefix + '_meas_standard', ''),
            'conditions_json': js(conditions), 'source_value_column_1based': value_index + 1,
        }


def target_name(row):
    text = row[9] + ' ' + row[10]
    text = re.sub(r'\S*(?:imidazol\w*|imidazolin\w*|carbodiimid\w*|maleimid\w*|citraconimid\w*)', '', text, flags=re.I)
    if re.search(r'\b(?:co[- ]?)?poly[a-z-]*imide|ポリ[^/;]*イミド', text, re.I):
        return True
    # Inspect balanced polymer-name brackets, so an imide salt/additive after
    # poly(ethylene oxide), for example, cannot identify that polymer as a PI.
    for start in re.finditer(r'\bpoly\s*[\[(]', text, re.I):
        depth, stop = 1, start.end()
        while stop < len(text) and depth:
            if text[stop] in '([':
                depth += 1
            elif text[stop] in ')]':
                depth -= 1
            stop += 1
        if re.search(r'imid(?:e|o)', text[start.end():stop], re.I):
            return True
    return False


def annotate_sample(row, units, pid_map):
    ids, complete, mapping_source, expression, groups, blend = map_components(row, units, pid_map)
    # composition_id may describe only one polymer within a blend.
    complete = complete and not blend
    vals = composition_values(row[38])
    aligned = mapping_source == 'composition_id' and len(groups) == len(ids) and all(CU_RE.fullmatch(g) for g in groups) and vals is not None and len(vals) == len(ids)
    active = [i for i in range(len(ids)) if not aligned or vals[i] > 0]
    active_ids = [ids[i] for i in active]
    active_units = [units[c] for c in active_ids if c in units]
    mainchain = any(u['backbone_imide'] for u in active_units)
    cyclic = any(u['cyclic_imide'] for u in active_units)
    named = target_name(row)
    has_zero_imide = aligned and not cyclic and any(units.get(c, {}).get('cyclic_imide') for c in ids)
    if mainchain:
        scope = 'polyimide'
        evidence = 'backbone_cyclic_imide_in_SMILES'
    elif named and not has_zero_imide:
        # A complete valid non-imide structure contradicts the name; keep it for review.
        contradicted = complete and active_units and all(u['smiles_valid'] for u in active_units)
        scope = 'review' if contradicted else 'polyimide'
        evidence = 'name_structure_conflict' if contradicted else 'polyimide_name_structure_unresolved'
    elif cyclic:
        scope, evidence = 'related_imide', 'cyclic_imide_without_backbone_N_path'
    else:
        scope, evidence = 'excluded', 'zero_fraction_imide' if has_zero_imide else 'non_polyimide_name_match'
    category = row[15]
    if blend:
        polymer_kind = 'blend_or_composite'
    elif aligned and len(active_ids) == 1:
        polymer_kind = 'homopolymer'
    elif len(ids) > 1:
        polymer_kind = 'copolymer'
    elif complete and len(ids) == 1:
        polymer_kind = 'homopolymer'
    elif re.search(r'block|random|statistical|graft|alternating|periodic', category, re.I) or re.search(r'copoly|co-poly', row[9] + ' ' + row[10], re.I):
        polymer_kind = 'copolymer'
    else:
        polymer_kind = 'unknown'
    if polymer_kind == 'homopolymer':
        sequence, sequence_source = 'not_applicable', 'single_active_repeat_unit'
    else:
        labels = [v for v in ('block', 'random', 'statistical', 'alternating', 'graft', 'periodic') if re.search(r'\b' + v + r'\b', category, re.I)]
        sequence_source = 'copolymer_category'
        if not labels:
            name = row[9] + ' ' + row[10]
            labels = [v for v in ('block', 'random', 'statistical', 'alternating', 'graft', 'periodic') if re.search(r'\b' + v + r'\b', name, re.I)]
            sequence_source = 'explicit_polymer_name' if labels else 'unknown'
        sequence = '+'.join(labels) if labels else 'unknown'
        if labels and re.search('unspecified', category, re.I):
            sequence += '+unspecified'
    valid_complete = complete and bool(active_ids) and all(c in units and units[c]['smiles_valid'] for c in active_ids)
    known_smiles = '.'.join(u['canonical_smiles'] for u in active_units if u['smiles_valid'])
    fractions, mol_ratio, mol_basis, derived = [], '', '', ''
    kind = unit_kind(row[37])
    if kind == 'mol' and row[38]:
        mol_ratio = ':'.join(split_top(row[38]))
        mol_basis = 'reported_CU_mol_composition' if aligned else 'reported_CU_mol_composition_alignment_unresolved'
        if aligned:
            fractions = [x / sum(vals) for x in vals]
    elif kind == 'weight' and aligned and complete and all(units[c]['repeat_unit_mw_g_mol'] for c in ids):
        mole_counts = [x / units[c]['repeat_unit_mw_g_mol'] for x, c in zip(vals, ids)]
        fractions = [x / sum(mole_counts) for x in mole_counts]
        derived = ':'.join(fmt(x) for x in fractions)
        mol_basis = 'calculated_repeat_unit_moles_from_CU_weight_composition'
    elif polymer_kind == 'homopolymer' and len(ids) == 1:
        fractions, mol_ratio, mol_basis = [1.0], '1', 'single_repeat_unit'
    feed_mol = ':'.join(row[36].split('/')) if unit_kind(row[35]) == 'mol' else ''
    material = 'neat_polymer' if row[14] == 'Neat resin' and not blend and not any(row[i] for i in (20, 24, 27)) else 'modified_or_composite_or_unspecified'
    notes = []
    if not valid_complete:
        notes.append('Complete validated SMILES unavailable; known components are separate.')
    if len(ids) > 1:
        notes.append('Dot SMILES lists disconnected repeat units; sequence and composition are separate fields.')
    if derived:
        notes.append('Calculated mol fractions count the listed repeat units, not block molecules or block lengths.')
    if row[37] and kind == 'unknown':
        notes.append('Composition unit does not explicitly specify moles or convertible weight fractions.')
    if feed_mol and not mol_ratio:
        notes.append('Feed mol ratio is not a measured polymer repeat-unit mol ratio.')
    base = {
        'sample_id': row[1], 'registry_pid': row[4], 'polymer_name': row[10],
        'reference_polymer_name': row[9], 'reference_no': row[2],
        'sample_identification_id': row[3], 'sample_fabrication_id': row[30],
        'scope': scope, 'polyimide_evidence': evidence, 'polymer_kind': polymer_kind,
        'sequence_type': sequence, 'sequence_source': sequence_source, 'copolymer_category_raw': category,
        'smiles': known_smiles if valid_complete else '',
        'smiles_representation': 'single_repeat_unit' if valid_complete and len(active_ids) == 1 else 'disconnected_component_repeat_units' if valid_complete else 'unresolved',
        'smiles_complete': valid_complete, 'known_component_smiles': known_smiles,
        'component_ids': '/'.join(ids), 'active_component_ids': '/'.join(active_ids),
        'component_mapping_source': mapping_source, 'component_expression_raw': expression,
        'component_ids_complete': complete,
        'composition_mol_ratio_reported': mol_ratio, 'composition_mol_basis': mol_basis,
        'composition_mol_ratio_calculated': derived, 'composition_mole_fractions_json': js(fractions) if fractions else '',
        'composition_id_raw': row[34], 'composition_unit_raw': row[37], 'composition_value_raw': row[38],
        'reactant_names_raw': row[31], 'feed_mol_ratio_reported': feed_mol,
        'feed_composition_unit_raw': row[35], 'feed_composition_value_raw': row[36],
        'material_type_raw': row[14], 'material_group': material, 'is_blend': blend,
        'blend_id_raw': row[7], 'blend_composition_raw': row[39],
        'blend_composition_unit_raw': row[40], 'blend_composition_value_raw': row[41],
        'filler_raw': row[20], 'additive_raw': row[24], 'other_component_raw': row[27],
        'filler_unit_raw': row[21], 'filler_value_raw': row[22], 'filler_shape_raw': row[23],
        'additive_unit_raw': row[25], 'additive_value_raw': row[26],
        'other_component_unit_raw': row[28], 'other_component_value_raw': row[29],
        'sample_shape_raw': row[124], 'sample_state_raw': row[125],
        'shape_construction_raw': row[16], 'notes': ' '.join(notes),
    }
    components = []
    for i, cu in enumerate(ids):
        u = units.get(cu, {})
        components.append({
            'sample_id': row[1], 'component_order': i + 1, 'component_id': cu,
            'active': i in active, 'component_registry_pid': u.get('registry_pid', ''),
            'smiles': u.get('canonical_smiles', ''), 'smiles_valid': u.get('smiles_valid', False),
            'backbone_imide': u.get('backbone_imide', False), 'cyclic_imide': u.get('cyclic_imide', False),
            'repeat_unit_mw_g_mol': u.get('repeat_unit_mw_g_mol', ''),
            'mole_fraction': fractions[i] if len(fractions) == len(ids) else '',
            'composition_value_raw': fmt(vals[i]) if aligned else '',
            'composition_unit_raw': row[37].split('/')[i] if aligned and len(row[37].split('/')) == len(ids) else '',
            'excel_cell': u.get('excel_cell', ''), 'mapping_source': mapping_source,
        })
    return base, components


def scalar_value(measurements, prop):
    vals = [m for m in measurements if m['property'] == prop]
    # Never select one measurement from several conditions or average them.
    if len(vals) == 1 and vals[0]['value_min'] != '' and vals[0]['value_min'] == vals[0]['value_max']:
        return vals[0]['value_min'], vals[0]['qualifier']
    return '', ''


def write_tsv(path, records, columns=None):
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(path, sep='\t', index=False, encoding='utf-8-sig')
    return df


def structure_index(samples, measurements):
    """Index complete, unmodified polymers without manufacturing Tg/E pairs."""
    groups, sid_to_group = {}, {}
    for s in samples:
        s['structure_group_id'] = ''
        if not s['smiles_complete'] or s['material_group'] != 'neat_polymer':
            continue
        if s['polymer_kind'] == 'homopolymer':
            key = js([s['smiles'], 'homopolymer'])
        else:
            composition = s['composition_mole_fractions_json'] or s['composition_mol_ratio_reported']
            # Unknown compositions cannot be identified across different fabrications.
            key = js([s['smiles'], s['sequence_type'], composition or s['sample_fabrication_id']])
        if key not in groups:
            groups[key] = {'id': f'G{len(groups) + 1:06d}', 'samples': [], 'measurements': []}
        g = groups[key]
        g['samples'].append(s)
        s['structure_group_id'] = g['id']
        sid_to_group[s['sample_id']] = g
    for m in measurements:
        g = sid_to_group.get(m['sample_id'])
        m['structure_group_id'] = g['id'] if g else ''
        if g:
            g['measurements'].append(m)
    rows = []
    for g in groups.values():
        ss, ms = g['samples'], g['measurements']
        first = ss[0]
        def labels(prop):
            values = []
            for m in ms:
                if m['property'] != prop:
                    continue
                label = m['qualifier'] + ' ' if m['qualifier'] else ''
                label += str(m['value_min']) if m['value_min'] != '' else m['value_min_raw']
                if m['value_min_raw'] != m['value_max_raw']:
                    label += '..' + (str(m['value_max']) if m['value_max'] != '' else m['value_max_raw'])
                label += ' ' + (m['unit'] or m['unit_raw'])
                if label not in values:
                    values.append(label)
            return js(values)
        homo = first['polymer_kind'] == 'homopolymer'
        rows.append({
            'structure_group_id': g['id'], 'smiles': first['smiles'],
            'polymer_kind': first['polymer_kind'], 'sequence_type': first['sequence_type'],
            'component_ids': first['active_component_ids'] if homo else first['component_ids'],
            'composition_mole_fractions_json': '[1.0]' if homo else first['composition_mole_fractions_json'],
            'composition_mol_ratio_reported': '1' if homo else first['composition_mol_ratio_reported'],
            'composition_mol_ratio_calculated': '' if homo else first['composition_mol_ratio_calculated'],
            'composition_mol_bases_json': js(list(dict.fromkeys(s['composition_mol_basis'] for s in ss))),
            'sample_count': len(ss), 'measurement_count': len(ms),
            'has_Tg': any(m['property'] == 'Tg' for m in ms),
            'has_tensile_modulus': any(m['property'] == 'tensile_modulus' for m in ms),
            'has_any_modulus': any(m['property'] != 'Tg' for m in ms),
            'Tg_values_unique_json': labels('Tg'),
            'tensile_modulus_values_unique_json': labels('tensile_modulus'),
            'modulus_types_json': js(list(dict.fromkeys(m['property'] for m in ms if m['property'] != 'Tg'))),
            'registry_pids_json': js(list(dict.fromkeys(s['registry_pid'] for s in ss))),
            'reference_names_json': js(list(dict.fromkeys(s['reference_polymer_name'] for s in ss))),
            'sample_ids_json': js([s['sample_id'] for s in ss]),
            'measurement_ids_json': js([m['measurement_id'] for m in ms]),
            'grouping_scope': 'structure_and_known_composition' if homo or first['composition_mole_fractions_json'] or first['composition_mol_ratio_reported'] else 'same_fabrication_composition_unknown',
            'notes': 'Overview across samples; values retain their original sample/condition in Measurements. No averaged or artificially paired values.',
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tsv', type=Path, default=Path('data_20240322.tsv'))
    ap.add_argument('--xlsx', type=Path, default=Path('01.smilesList.xlsx'))
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    units, pid_map = read_units(args.xlsx)
    print(f'Validated {len(units)} Excel repeat units.', flush=True)
    groups, source_rows, total = {}, {}, 0
    with args.tsv.open(encoding='cp932', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        jp, en = next(reader), next(reader)
        for line, row in enumerate(reader, 3):
            total += 1
            if len(row) != len(en):
                raise ValueError(f'Column mismatch at logical TSV row {line}')
            ids, _, _, _, _, _ = map_components(row, units, pid_map)
            structural = any(units.get(c, {}).get('cyclic_imide') for c in ids)
            if not structural and not re.search(r'imid|イミド', row[9] + ' ' + row[10], re.I):
                continue
            sid = row[1]
            if sid not in groups:
                groups[sid] = {'row': row, 'lines': [], 'sidps': [], 'measurements': {}}
            g = groups[sid]
            # Sample identity and composition must agree before grouping any measurements.
            for i in list(range(4, 42)) + [124, 125]:
                if row[i] != g['row'][i]:
                    raise ValueError(f'Conflicting sample metadata: {sid}, column {i + 1}')
            g['lines'].append(line)
            g['sidps'].append(row[0])
            for m in measurements_from_row(row, en):
                key = js(m)
                if key not in g['measurements']:
                    m['source_tsv_rows'] = []
                    m['source_sidps'] = []
                    g['measurements'][key] = m
                g['measurements'][key]['source_tsv_rows'].append(line)
                g['measurements'][key]['source_sidps'].append(row[0])
    print(f'Read {total} TSV data rows; assembling {len(groups)} candidate samples.', flush=True)
    samples, components, measurements, related, review, exclusions, no_property = [], [], [], [], [], [], []
    expected_main_measurements = set()
    for sid, g in groups.items():
        base, comps = annotate_sample(g['row'], units, pid_map)
        ms = list(g['measurements'].values())
        base['has_Tg'] = any(m['property'] == 'Tg' for m in ms)
        base['has_modulus'] = any(m['property'] != 'Tg' for m in ms)
        base['has_tensile_modulus'] = any(m['property'] == 'tensile_modulus' for m in ms)
        base['Tg_C'], base['Tg_qualifier'] = scalar_value(ms, 'Tg')
        base['tensile_modulus_GPa'], base['tensile_modulus_qualifier'] = scalar_value(ms, 'tensile_modulus')
        base['Tg_measurement_count'] = sum(m['property'] == 'Tg' for m in ms)
        base['modulus_measurement_count'] = sum(m['property'] != 'Tg' for m in ms)
        base['source_tsv_rows'] = js(g['lines'])
        base['source_sidps'] = js(g['sidps'])
        base['measurements_json'] = js([{k: v for k, v in m.items() if k not in ('source_tsv_rows', 'source_sidps')} for m in ms])
        if base['scope'] == 'polyimide' and ms:
            samples.append(base)
            components.extend(comps)
            for m in ms:
                record = {k: base[k] for k in (
                    'sample_id', 'registry_pid', 'polymer_name', 'reference_polymer_name', 'reference_no',
                    'polymer_kind', 'sequence_type', 'smiles', 'smiles_complete', 'component_ids',
                    'active_component_ids', 'composition_mol_ratio_reported', 'composition_mol_ratio_calculated',
                    'composition_mole_fractions_json', 'composition_mol_basis', 'feed_mol_ratio_reported',
                    'material_group', 'material_type_raw', 'is_blend')}
                record.update(m)
                record['source_tsv_rows'] = js(m['source_tsv_rows'])
                record['source_sidps'] = js(m['source_sidps'])
                measurements.append(record)
        elif base['scope'] == 'polyimide':
            no_property.append(base)
        elif base['scope'] == 'related_imide':
            related.append(base)
        elif base['scope'] == 'review':
            review.append(base)
        else:
            exclusions.append(base)
    # Stable IDs: distinct property/conditions/value records, never a Tg x E Cartesian join.
    for i, m in enumerate(measurements, 1):
        m['measurement_id'] = f'M{i:06d}'
    for i, s in enumerate(samples, 1):
        s['dataset_sample_id'] = f'S{i:06d}'
    # Allocate original IDs before exclusion so existing model provenance stays valid.
    material_exclusions = []
    for table_name, records in [('Samples', samples), ('Related_imides', related),
                                ('Review_conflicts', review), ('Excluded', exclusions),
                                ('No_target_property', no_property)]:
        material_exclusions.extend({
            'sample_id': s['sample_id'], 'original_table': table_name,
            'material_type_raw': s['material_type_raw'], 'material_group': s['material_group'],
            'reason': 'Exclude blends, fillers, additives and materials not confirmed as neat resin.'
        } for s in records if s['material_group'] != 'neat_polymer')
    removed_main = sum(s['material_group'] != 'neat_polymer' for s in samples)
    removed_measurements = sum(m['material_group'] != 'neat_polymer' for m in measurements)
    samples = [s for s in samples if s['material_group'] == 'neat_polymer']
    measurements = [m for m in measurements if m['material_group'] == 'neat_polymer']
    retained_ids = {s['sample_id'] for s in samples}
    components = [c for c in components if c['sample_id'] in retained_ids]
    related, review, exclusions, no_property = [
        [s for s in records if s['material_group'] == 'neat_polymer']
        for records in (related, review, exclusions, no_property)]
    write_tsv(out / 'material_exclusion_audit.tsv', material_exclusions)
    def below_tensile_floor(m):
        upper = number(m.get('value_max'))
        return (m.get('property') == 'tensile_modulus' and m.get('unit') == 'GPa'
                and upper is not None and upper <= 0.01)
    low_moduli = [m for m in measurements if below_tensile_floor(m)]
    write_tsv(out / 'tensile_floor_exclusion_audit.tsv', [
        {'sample_id': m['sample_id'], 'measurement_id': m['measurement_id'],
         'property_id': m['property_id'], 'reason': 'tensile_modulus <= 0.01 GPa'}
        for m in low_moduli])
    measurements = [m for m in measurements if not below_tensile_floor(m)]
    by_sample = collections.defaultdict(list)
    for m in measurements:
        by_sample[m['sample_id']].append(m)
    kept_samples = []
    for s in samples:
        ms = by_sample[s['sample_id']]
        s['has_Tg'] = any(m['property'] == 'Tg' for m in ms)
        s['has_modulus'] = any(m['property'] != 'Tg' for m in ms)
        s['has_tensile_modulus'] = any(m['property'] == 'tensile_modulus' for m in ms)
        s['Tg_measurement_count'] = sum(m['property'] == 'Tg' for m in ms)
        s['modulus_measurement_count'] = sum(m['property'] != 'Tg' for m in ms)
        s['Tg_C'], s['Tg_qualifier'] = scalar_value(ms, 'Tg')
        s['tensile_modulus_GPa'], s['tensile_modulus_qualifier'] = scalar_value(ms, 'tensile_modulus')
        s['measurements_json'] = js([m for m in json.loads(s['measurements_json'])
                                      if not below_tensile_floor(m)])
        (kept_samples if ms else no_property).append(s)
    samples = kept_samples
    for records in (related, review, exclusions, no_property):
        for s in records:
            ms = [m for m in json.loads(s['measurements_json']) if not below_tensile_floor(m)]
            s['measurements_json'] = js(ms)
            s['has_Tg'] = any(m['property'] == 'Tg' for m in ms)
            s['has_modulus'] = any(m['property'] != 'Tg' for m in ms)
            s['has_tensile_modulus'] = any(m['property'] == 'tensile_modulus' for m in ms)
            s['Tg_measurement_count'] = sum(m['property'] == 'Tg' for m in ms)
            s['modulus_measurement_count'] = sum(m['property'] != 'Tg' for m in ms)
            s['Tg_C'], s['Tg_qualifier'] = scalar_value(ms, 'Tg')
            s['tensile_modulus_GPa'], s['tensile_modulus_qualifier'] = scalar_value(ms, 'tensile_modulus')
    retained_ids = {s['sample_id'] for s in samples}
    components = [c for c in components if c['sample_id'] in retained_ids]
    structure_rows = structure_index(samples, measurements)
    sf = write_tsv(out / 'polyimide_structure_composition_summary.tsv', structure_rows)
    columns = list(samples[0]) if samples else []
    # Put the convenient values next to SMILES in spreadsheet/TSV views.
    front = ['dataset_sample_id', 'sample_id', 'registry_pid', 'polymer_kind', 'sequence_type', 'smiles',
             'smiles_complete', 'Tg_C', 'Tg_qualifier', 'tensile_modulus_GPa', 'tensile_modulus_qualifier',
             'composition_mol_ratio_reported', 'composition_mol_ratio_calculated', 'composition_mol_basis',
             'composition_mole_fractions_json', 'component_ids', 'feed_mol_ratio_reported']
    columns = front + [k for k in columns if k not in front]
    df = write_tsv(out / 'polyimide_samples.tsv', samples, columns)
    mf = write_tsv(out / 'polyimide_measurements.tsv', measurements)
    cf = write_tsv(out / 'polyimide_components.tsv', components)
    both = df[df.has_Tg & df.has_modulus]
    neat = df[(df.material_group == 'neat_polymer') & df.smiles_complete]
    unresolved = df[~df.smiles_complete]
    write_tsv(out / 'polyimide_both_Tg_modulus.tsv', both.to_dict('records'), columns)
    write_tsv(out / 'polyimide_neat_complete_smiles.tsv', neat.to_dict('records'), columns)
    rf = write_tsv(out / 'related_imide_samples.tsv', related, columns)
    vf = write_tsv(out / 'review_name_structure_conflicts.tsv', review, columns)
    ef = write_tsv(out / 'excluded_candidates.tsv', exclusions, columns)
    npf = write_tsv(out / 'polyimide_without_target_properties.tsv', no_property, columns)
    uf = write_tsv(out / 'repeat_units_used.tsv', [units[c] for c in sorted(set(cf.component_id)) if c in units])
    guide = write_tsv(out / 'column_guide.tsv', [{'column': k, '説明': v} for k, v in COLUMN_GUIDE.items()])
    write_tsv(out / 'source_columns.tsv', [{'column_1based': i + 1, 'japanese': j, 'english': e} for i, (j, e) in enumerate(zip(jp, en))])
    summary = {
        'material_policy': 'neat_polymer_only',
        'removed_non_neat_main_samples': removed_main,
        'removed_non_neat_measurements': removed_measurements,
        'tensile_modulus_min_exclusive_GPa': 0.01,
        'removed_low_tensile_measurements': len(low_moduli),
        'source_data_rows': total, 'excel_repeat_units': len(units), 'excel_invalid_smiles': sum(not u['smiles_valid'] for u in units.values()),
        'polyimide_samples_with_Tg_or_modulus': len(df), 'unique_registry_pid_strings': df.registry_pid.nunique(),
        'samples_with_Tg': int(df.has_Tg.sum()), 'samples_with_any_modulus': int(df.has_modulus.sum()),
        'samples_with_tensile_modulus': int(df.has_tensile_modulus.sum()), 'samples_with_both_Tg_and_any_modulus': len(both),
        'complete_smiles_samples': int(df.smiles_complete.sum()), 'unresolved_smiles_samples': len(unresolved),
        'neat_complete_smiles_samples': len(neat), 'polymer_kind': df.polymer_kind.value_counts().to_dict(),
        'neat_structure_composition_groups': len(sf),
        'neat_structure_groups_with_Tg_and_tensile_modulus': int((sf.has_Tg & sf.has_tensile_modulus).sum()),
        'copolymer_sequence_type': df[df.polymer_kind == 'copolymer'].sequence_type.value_counts().to_dict(),
        'copolymer_reported_mol_ratio': int(((df.polymer_kind == 'copolymer') & (df.composition_mol_ratio_reported != '')).sum()),
        'copolymer_calculated_mol_ratio': int(((df.polymer_kind == 'copolymer') & (df.composition_mol_ratio_calculated != '')).sum()),
        'copolymer_feed_mol_ratio': int(((df.polymer_kind == 'copolymer') & (df.feed_mol_ratio_reported != '')).sum()),
        'unique_measurements': len(mf), 'measurements_by_property': mf.property.value_counts().to_dict(),
        'related_imide_samples': len(rf), 'name_structure_conflicts': len(vf),
        'excluded_samples': len(ef), 'polyimide_without_target_properties': len(npf),
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    summary_rows = [{'項目': k, '値': js(v) if isinstance(v, dict) else v} for k, v in summary.items()]
    with pd.ExcelWriter(out / 'polyimide_Tg_modulus_SMILES.xlsx', engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_formulas': False, 'strings_to_urls': False}}) as writer:
        tables = [('Summary', pd.DataFrame(summary_rows)), ('Column_guide', guide), ('Structure_composition', sf), ('Samples', df), ('Both_Tg_modulus', both),
                  ('Neat_complete_SMILES', neat), ('Measurements', mf), ('Components', cf),
                  ('Unresolved_SMILES', unresolved), ('Related_imides', rf), ('Review_conflicts', vf),
                  ('No_target_property', npf), ('Excluded', ef), ('Repeat_units', uf)]
        for title, frame in tables:
            longest_cell = max((len(x) for col in frame.columns for x in frame[col] if isinstance(x, str)), default=0)
            if longest_cell > 32767:
                raise ValueError(f'Excel cell length limit exceeded: {title}. Full TSV retained.')
            frame.to_excel(writer, sheet_name=title, index=False)
            sheet = writer.sheets[title]
            sheet.freeze_panes(1, min(2, len(frame.columns)))
            if len(frame.columns):
                sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
                sheet.set_column(0, len(frame.columns) - 1, 20)
                for col, name in enumerate(frame.columns):
                    if 'smiles' in name or name in ('polymer_name', 'reference_polymer_name'):
                        sheet.set_column(col, col, 48)
    # Validation uses the finished artifacts, including the Excel round trip.
    assert len(df) == df.sample_id.nunique()
    assert all(Chem.MolFromSmiles(s) is not None for s in df.loc[df.smiles_complete, 'smiles'])
    assert not any(row['scope'] != 'polyimide' for row in samples)
    assert all(s['has_Tg'] or s['has_modulus'] for s in samples)
    for sample in samples:
        fs = json.loads(sample['composition_mole_fractions_json']) if sample['composition_mole_fractions_json'] else []
        if fs:
            assert len(fs) == len(sample['component_ids'].split('/'))
            assert abs(sum(fs) - 1) < 1e-10 and all(0 <= x <= 1 for x in fs)
    book = openpyxl.load_workbook(out / 'polyimide_Tg_modulus_SMILES.xlsx', read_only=True, data_only=True)
    assert book['Samples'].max_row == len(df) + 1
    assert book['Measurements'].max_row == len(mf) + 1
    book.close()
    fingerprints = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.tsv, args.xlsx)}
    (out / 'validation.json').write_text(json.dumps({'status': 'passed', 'input_sha256': fingerprints,
        'checks': ['sample IDs unique', 'complete SMILES parsed by RDKit', 'mole fractions aligned and normalized',
                   'sample metadata consistent across source rows', 'property records deduplicated with conditions and property IDs',
                   'Excel row counts agree with exported tables']}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
