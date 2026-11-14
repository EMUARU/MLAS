#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic.constraints

欄位約束系統：單欄位與 IF-THEN 條件約束之解析、前置過濾與後置過濾。

"""

import json

import numpy as np
import pandas as pd


def parse_column_constraints(json_str):
    """
    解析欄位約束 JSON 字串，支援兩種格式：
    
    舊格式（向後相容）- 純欄位約束:
        {"age": {"type": "range", "min": 18, "max": 65}}
    
    新格式 - 欄位約束 + 條件規則:
        {
          "columns": {"age": {"type": "range", "min": 18, "max": 65}},
          "conditionals": [
            {
              "when": {"column": "gender", "operator": "==", "value": "F"},
              "then": {"column": "age", "type": "range", "min": 18, "max": 65}
            }
          ]
        }
    
    Returns:
        dict: 統一格式 {"columns": {...}, "conditionals": [...]}
              空時返回空 dict {} 以便相容現有的 if 判斷
    """
    if not json_str or str(json_str).lower() in ('none', 'null', '{}', ''):
        return {}
    
    try:
        raw = json.loads(json_str)
        if not isinstance(raw, dict):
            return {}
        
        # 偵測格式：有 "columns" 或 "conditionals" key 視為新格式
        is_new_format = ('columns' in raw or 'conditionals' in raw)
        
        if is_new_format:
            columns_raw = raw.get('columns', {}) if isinstance(raw.get('columns'), dict) else {}
            conditionals_raw = raw.get('conditionals', []) if isinstance(raw.get('conditionals'), list) else []
        else:
            # 舊格式：整個 dict 都是 column constraints
            columns_raw = raw
            conditionals_raw = []
        
        # 解析單欄位約束
        valid_columns = {}
        for col, rule in columns_raw.items():
            parsed = _parse_single_constraint(rule)
            if parsed is not None:
                valid_columns[col] = parsed
        
        # 解析條件規則
        valid_conditionals = []
        for cond in conditionals_raw:
            parsed_cond = _parse_conditional(cond)
            if parsed_cond is not None:
                valid_conditionals.append(parsed_cond)
        
        # 如果都空就返回空 dict
        if not valid_columns and not valid_conditionals:
            return {}
        
        return {
            'columns': valid_columns,
            'conditionals': valid_conditionals
        }
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_single_constraint(rule):
    """解析單個欄位約束規則"""
    if not isinstance(rule, dict) or 'type' not in rule:
        return None
    
    ctype = rule['type'].lower()
    
    if ctype == 'range':
        if 'min' in rule or 'max' in rule:
            return {'type': 'range', 'min': rule.get('min'), 'max': rule.get('max')}
    elif ctype == 'min' and 'value' in rule:
        return {'type': 'min', 'value': rule['value']}
    elif ctype == 'max' and 'value' in rule:
        return {'type': 'max', 'value': rule['value']}
    elif ctype == 'fixed' and 'value' in rule:
        return {'type': 'fixed', 'value': rule['value']}
    elif ctype == 'options' and 'values' in rule and isinstance(rule['values'], list):
        return {'type': 'options', 'values': rule['values']}
    
    return None


def _parse_conditional(cond):
    """解析條件規則 IF (when) THEN (then)"""
    if not isinstance(cond, dict):
        return None
    if 'when' not in cond or 'then' not in cond:
        return None
    
    when = cond['when']
    then = cond['then']
    
    # 驗證 when 子句
    if not isinstance(when, dict) or 'column' not in when or 'operator' not in when:
        return None
    
    op = when['operator']
    valid_ops = ('==', '!=', '>=', '<=', '>', '<', 'in', 'not_in', 'between')
    if op not in valid_ops:
        return None
    
    if op in ('in', 'not_in'):
        if 'values' not in when or not isinstance(when['values'], list):
            return None
    elif op == 'between':
        if 'min' not in when or 'max' not in when:
            return None
    else:
        if 'value' not in when:
            return None
    
    # 驗證 then 子句（必須有 column，其他用 _parse_single_constraint）
    if not isinstance(then, dict) or 'column' not in then:
        return None
    
    then_constraint = _parse_single_constraint(then)
    if then_constraint is None:
        return None
    
    return {
        'when': when,
        'then': {'column': then['column'], **then_constraint}
    }


def _evaluate_when_clause(df, when):
    """評估 when 子句，返回 boolean Series（True 表示該行符合條件）"""
    col = when['column']
    op = when['operator']
    
    if col not in df.columns:
        # 欄位不存在，視為都不符合條件
        return pd.Series(False, index=df.index)
    
    series = df[col]
    
    if op == '==':
        return series == when['value']
    elif op == '!=':
        return series != when['value']
    elif op == '>=':
        return series >= when['value']
    elif op == '<=':
        return series <= when['value']
    elif op == '>':
        return series > when['value']
    elif op == '<':
        return series < when['value']
    elif op == 'in':
        return series.isin(when['values'])
    elif op == 'not_in':
        return ~series.isin(when['values'])
    elif op == 'between':
        return (series >= when['min']) & (series <= when['max'])
    
    return pd.Series(False, index=df.index)


def _evaluate_then_constraint(df, then):
    """評估 then 子句，返回 boolean Series（True 表示該行符合 then 約束）"""
    col = then['column']
    ctype = then['type']
    
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    
    series = df[col]
    mask = pd.Series(True, index=df.index)
    
    if ctype == 'range':
        if then.get('min') is not None:
            mask &= (series >= then['min'])
        if then.get('max') is not None:
            mask &= (series <= then['max'])
    elif ctype == 'min':
        mask &= (series >= then['value'])
    elif ctype == 'max':
        mask &= (series <= then['value'])
    elif ctype == 'fixed':
        mask &= (series == then['value'])
    elif ctype == 'options':
        mask &= series.isin(then['values'])
    
    # NaN 視為通過
    mask = mask | series.isna()
    return mask


def prefilter_data_by_constraints(df, constraints, min_rows=10, label_column=None):
    """
    訓練前過濾原始資料，確保訓練資料 100% 符合所有約束。
    
    Args:
        df: 原始 DataFrame
        constraints: parse_column_constraints() 的輸出（統一格式）
        min_rows: 過濾後最少需保留的行數
        label_column: 可選，若指定（grouped 模式用），會額外檢查過濾後是否仍涵蓋
                     所有原始類別。若有任何類別完全消失或剩餘類別 < 2，則觸發 fallback。
        
    Returns:
        tuple: (filtered_df, filter_report)
    """
    if not constraints:
        return df, {}
    
    # 統一格式處理
    columns_dict = constraints.get('columns', {}) if isinstance(constraints, dict) else {}
    conditionals = constraints.get('conditionals', []) if isinstance(constraints, dict) else []
    
    # 向後相容：如果 constraints 不是統一格式（舊呼叫方式），整個視為 columns
    if 'columns' not in constraints and 'conditionals' not in constraints:
        columns_dict = constraints
        conditionals = []
    
    original_rows = len(df)
    mask = pd.Series(True, index=df.index)
    report = {'columns': {}, 'conditionals': []}
    
    # === 1. 套用單欄位約束 ===
    for col, rule in columns_dict.items():
        if col not in df.columns:
            report['columns'][col] = {'status': 'skipped', 'reason': 'column_not_found'}
            continue
        
        ctype = rule['type']
        col_mask = pd.Series(True, index=df.index)
        
        if ctype == 'range':
            if rule.get('min') is not None:
                col_mask &= (df[col] >= rule['min'])
            if rule.get('max') is not None:
                col_mask &= (df[col] <= rule['max'])
        elif ctype == 'min':
            col_mask &= (df[col] >= rule['value'])
        elif ctype == 'max':
            col_mask &= (df[col] <= rule['value'])
        elif ctype == 'fixed':
            col_mask &= (df[col] == rule['value'])
        elif ctype == 'options':
            col_mask &= df[col].isin(rule['values'])
        
        # NaN 保留
        col_mask = col_mask | df[col].isna()
        
        removed = int((~col_mask).sum())
        report['columns'][col] = {
            'status': 'applied',
            'type': ctype,
            'rows_removed': removed
        }
        mask &= col_mask
    
    # === 2. 套用條件規則 ===
    # 條件規則邏輯：對每行 row，如果 when 條件成立，則 then 約束必須滿足
    # 等價於：keep row if NOT(when AND NOT then) = keep row if (NOT when) OR (then)
    for idx, cond in enumerate(conditionals):
        when_mask = _evaluate_when_clause(df, cond['when'])
        then_mask = _evaluate_then_constraint(df, cond['then'])
        
        # 違反條件的行：when 成立 但 then 不成立
        violation_mask = when_mask & (~then_mask)
        cond_mask = ~violation_mask
        
        removed = int(violation_mask.sum())
        triggered = int(when_mask.sum())
        report['conditionals'].append({
            'index': idx,
            'when': cond['when'],
            'then': cond['then'],
            'rows_triggered': triggered,
            'rows_removed': removed
        })
        mask &= cond_mask
    
    filtered_df = df[mask].copy()
    
    # 安全檢查 1：過濾後資料太少則回退
    if len(filtered_df) < min_rows:
        report['_warning'] = (
            f'Pre-filter result too few rows ({len(filtered_df)}), '
            f'falling back to original data ({original_rows} rows). '
            f'Constraints will still be applied post-generation.'
        )
        return df, report

    # 安全檢查 2：grouped 模式 label 多樣性檢查
    # 若指定 label_column，過濾後必須仍涵蓋所有原始類別 且 至少 2 個類別
    # 否則 grouped 訓練會直接失敗（grouped 至少需 2 個類別）
    if label_column and label_column in df.columns and label_column != 'none':
        original_labels = set(df[label_column].dropna().unique())
        filtered_labels = set(filtered_df[label_column].dropna().unique())
        missing_labels = original_labels - filtered_labels

        if len(filtered_labels) < 2 or missing_labels:
            report['_warning'] = (
                f'Pre-filter eliminated entire label class(es) {sorted(str(x) for x in missing_labels)}. '
                f'Original labels: {sorted(str(x) for x in original_labels)}, '
                f'after filter: {sorted(str(x) for x in filtered_labels)}. '
                f'Falling back to original data ({original_rows} rows) to preserve all label classes. '
                f'Constraints will still be enforced post-generation.'
            )
            report['_label_fallback'] = {
                'original_labels': sorted(str(x) for x in original_labels),
                'filtered_labels': sorted(str(x) for x in filtered_labels),
                'missing_labels': sorted(str(x) for x in missing_labels)
            }
            return df, report

    report['_summary'] = {
        'original_rows': original_rows,
        'filtered_rows': len(filtered_df),
        'rows_removed': original_rows - len(filtered_df)
    }
    
    return filtered_df, report


def postfilter_synthetic_by_constraints(synthetic_df, constraints):
    """
    生成後過濾合成資料，保證 100% 合規。
    
    對單欄位數值約束：使用 np.clip 保留所有行
    對單欄位類別約束：行過濾
    對條件規則：行過濾（無法 clip，因為涉及多欄位邏輯）
    
    Returns:
        tuple: (filtered_df, filter_report)
    """
    if not constraints:
        return synthetic_df, {}
    
    columns_dict = constraints.get('columns', {}) if isinstance(constraints, dict) else {}
    conditionals = constraints.get('conditionals', []) if isinstance(constraints, dict) else []
    
    # 向後相容
    if 'columns' not in constraints and 'conditionals' not in constraints:
        columns_dict = constraints
        conditionals = []
    
    df = synthetic_df.copy()
    original_rows = len(df)
    report = {'columns': {}, 'conditionals': []}
    
    # === 1. 單欄位約束 ===
    for col, rule in columns_dict.items():
        if col not in df.columns:
            report['columns'][col] = {'status': 'skipped', 'reason': 'column_not_found'}
            continue
        
        ctype = rule['type']
        
        if ctype in ('range', 'min', 'max') and pd.api.types.is_numeric_dtype(df[col]):
            before_min = float(df[col].min()) if len(df) > 0 else None
            before_max = float(df[col].max()) if len(df) > 0 else None
            
            if ctype == 'range':
                df[col] = df[col].clip(lower=rule.get('min'), upper=rule.get('max'))
            elif ctype == 'min':
                df[col] = df[col].clip(lower=rule['value'])
            elif ctype == 'max':
                df[col] = df[col].clip(upper=rule['value'])
            
            report['columns'][col] = {
                'status': 'clipped',
                'type': ctype,
                'before_range': [before_min, before_max],
                'after_range': [float(df[col].min()), float(df[col].max())] if len(df) > 0 else None
            }
        elif ctype == 'fixed':
            before = len(df)
            df = df[df[col] == rule['value']].copy()
            report['columns'][col] = {
                'status': 'filtered', 'type': 'fixed',
                'rows_before': before, 'rows_after': len(df)
            }
        elif ctype == 'options':
            before = len(df)
            df = df[df[col].isin(rule['values'])].copy()
            report['columns'][col] = {
                'status': 'filtered', 'type': 'options',
                'rows_before': before, 'rows_after': len(df)
            }
    
    # === 2. 條件規則 ===
    for idx, cond in enumerate(conditionals):
        if len(df) == 0:
            break
        
        when_mask = _evaluate_when_clause(df, cond['when'])
        then_mask = _evaluate_then_constraint(df, cond['then'])
        
        # 違反：when 成立但 then 不成立
        violation_mask = when_mask & (~then_mask)
        keep_mask = ~violation_mask
        
        before = len(df)
        df = df[keep_mask].copy()
        report['conditionals'].append({
            'index': idx,
            'when': cond['when'],
            'then': cond['then'],
            'rows_removed': before - len(df)
        })
    
    report['_summary'] = {
        'original_rows': original_rows,
        'final_rows': len(df),
        'rows_removed': original_rows - len(df)
    }
    
    return df, report


# ============================================================
# 主程式
# ============================================================
