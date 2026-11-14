import sys
import json
import traceback
import pandas as pd
import numpy as np


def load_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print(json.dumps({
                "status": "error",
                "message": "Support .csv or .xlsx files only.",
            }))
            sys.exit(1)
        return df
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        sys.exit(1)


def apply_rules(df: pd.DataFrame, rules_json):
    """
    依使用者規則處理資料。
    支援之 expect_type:
      - drop_feature: 整欄刪除
      - not_missing: 處理缺失值
      - condition: 條件過濾
      - encode_label: 將類別字串編碼為數值 (Label Encoding)
    """
    modified_columns = set()
    rows_affected = set()
    skipped_rules = []

    for rule in rules_json:
        col = rule.get("column")
        expect_type = rule.get("expect_type")
        fallback_type = rule.get("fallback_type")

        if col not in df.columns:
            skipped_rules.append(f"{col}: column not found in dataframe")
            continue

        # ==========================================
        # Label Encoding 處理 (本研究新增)
        # ==========================================
        if expect_type == "encode_label":
            try:
                # 若已是數值類型,提示並跳過
                if pd.api.types.is_numeric_dtype(df[col]):
                    skipped_rules.append(
                        f"{col}: already numeric, encoding skipped"
                    )
                    continue
                # 字串轉成 category code
                # NaN 會自動編成 -1
                categorical = pd.Categorical(df[col].astype(str))
                df[col] = categorical.codes.astype(int)
                modified_columns.add(col)
            except Exception as e:
                skipped_rules.append(f"{col}: label encoding failed - {e}")
            continue

        # 處理整欄刪除
        if expect_type == "drop_feature":
            df.drop(columns=[col], inplace=True, errors="ignore")
            modified_columns.add(col)
            continue
        # 找出不符合的列
        elif expect_type == "not_missing":
            condition = df[col].isna()
        elif expect_type == "condition":
            cond = rule.get("expect_condition")
            val = rule.get("expect_value")
            try:
                val = float(val)
            except (TypeError, ValueError):
                skipped_rules.append(f"{col}: cannot parse expect_value '{rule.get('expect_value')}' as float")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                skipped_rules.append(
                    f"{col}: 'condition' rule requires numeric dtype, got {df[col].dtype}"
                )
                continue

            if cond == ">=":
                condition = ~(df[col] >= val)
            elif cond == "<=":
                condition = ~(df[col] <= val)
            elif cond == "!=":
                condition = ~(df[col] != val)
            else:
                skipped_rules.append(f"{col}: unsupported condition '{cond}'")
                continue
        else:
            skipped_rules.append(f"{col}: unsupported expect_type '{expect_type}'")
            continue

        # 處理 fallback
        if fallback_type == "drop":
            drop_indices = condition[condition].index
            df.drop(index=drop_indices, inplace=True)
            rows_affected.update(drop_indices.tolist())
            if len(drop_indices) > 0:
                modified_columns.add(col)
        elif fallback_type == "skip":
            continue
        else:
            fallback_value = None
            try:
                if fallback_type == "custom":
                    fallback_value = rule.get("fallback_value")
                elif fallback_type == "min":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        skipped_rules.append(
                            f"{col}: fallback 'min' requires numeric dtype, got {df[col].dtype}"
                        )
                        continue
                    fallback_value = df[col].min()
                elif fallback_type == "max":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        skipped_rules.append(
                            f"{col}: fallback 'max' requires numeric dtype, got {df[col].dtype}"
                        )
                        continue
                    fallback_value = df[col].max()
                elif fallback_type == "mean":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        skipped_rules.append(
                            f"{col}: fallback 'mean' requires numeric dtype, got {df[col].dtype}"
                        )
                        continue
                    fallback_value = df[col].mean()
                elif fallback_type == "median":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        skipped_rules.append(
                            f"{col}: fallback 'median' requires numeric dtype, got {df[col].dtype}"
                        )
                        continue
                    fallback_value = df[col].median()
                elif fallback_type == "mode":
                    mode_series = df[col].mode()
                    fallback_value = mode_series.iloc[0] if not mode_series.empty else None
                else:
                    skipped_rules.append(f"{col}: unsupported fallback_type '{fallback_type}'")
                    continue
            except Exception as e:
                skipped_rules.append(f"{col}: failed to compute {fallback_type} - {e}")
                continue

            if fallback_value is not None:
                affected_rows = condition[condition].index.tolist()
                if affected_rows:
                    try:
                        df.loc[condition, col] = fallback_value
                        modified_columns.add(col)
                        rows_affected.update(affected_rows)
                    except Exception as e:
                        skipped_rules.append(f"{col}: failed to assign fallback value - {e}")
                        continue

    return list(modified_columns), len(rows_affected), skipped_rules


def save_file(df: pd.DataFrame, file_path: str):
    try:
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        else:
            raise ValueError("Unsupported file format for saving.")
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Failed to save file: {e}",
        }))
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python preprocess.py <file_path> <rules_json>",
        }))
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        rules_json = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error",
            "message": f"Invalid rules JSON: {e}",
        }))
        sys.exit(1)

    df = load_file(file_path)
    df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}, inplace=True)

    try:
        modified_columns, affected_count, skipped_rules = apply_rules(df, rules_json)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"apply_rules failed: {e}",
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)

    save_file(df, file_path)

    message = f"Modified {len(modified_columns)} columns across {affected_count} rows."
    if skipped_rules:
        message += f" {len(skipped_rules)} rule(s) skipped."

    print(json.dumps({
        "status": "success",
        "modified_columns": modified_columns,
        "rows_affected": affected_count,
        "skipped_rules": skipped_rules,
        "message": message,
        "file_path": file_path,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()