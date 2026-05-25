import os
import glob
import json
import pandas as pd

# --- 設定路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
# 指向您存放重新評估後 JSON 檔的目錄
results_dir = os.path.join(workspace_root, "experiment_v2", "data", "results")

def analyze_results():
    # 搜尋所有符合 redteam_divi_results_*.json 的檔案
    json_pattern = os.path.join(results_dir, "redteam_divi_results_*.json")
    files = glob.glob(json_pattern)
    
    if not files:
        print(f"❌ 在 {results_dir} 找不到任何結果檔案。")
        return

    all_records = []

    print(f"🔍 正在分析 {len(files)} 個檔案...")

    for filepath in files:
        filename = os.path.basename(filepath)
        # 解析檔案名稱以取得攻擊手法名稱 (例如: redteam_divi_results_dan.json -> dan)
        method_name = filename.replace("redteam_divi_results_", "").replace(".json", "")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                # 取得情境與成功狀態 (由 re_evaluate_results.py 更新後的 success 欄位)
                scenario = item.get("scenario_desc", "Unknown Scenario")
                
                # 這裡假設 success 為 True/False (布林值)
                is_success = item.get("success", False)
                
                # 新增：取得模型資訊與分群資訊
                model = item.get("model", "Unknown Model")
                cluster = item.get("cluster", "Unassigned")

                all_records.append({
                    "Method": method_name,
                    "Scenario": scenario,
                    "Model": model,
                    "Cluster": cluster,
                    "Success": 1 if is_success else 0 
                })

        except Exception as e:
            print(f"⚠️ 讀取 {filename} 失敗: {e}")

    # 建立 DataFrame
    if not all_records:
        print("❌ 沒有讀取到任何數據記錄。")
        return

    df = pd.DataFrame(all_records)

    # ---------------------------------------------------------
    # 1. 攻擊手法成功率排名 (By Method)
    # ---------------------------------------------------------
    method_stats = df.groupby("Method")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    method_stats.columns = ['Attack Method', 'Total Attempts', 'Success Count', 'Success Rate']
    method_stats['Success Rate'] = (method_stats['Success Rate'] * 100).round(2)
    method_stats = method_stats.sort_values(by="Success Rate", ascending=False)

    print("\n" + "="*60)
    print("📊 各攻擊手法 (Attack Method) 越獄成功率排名")
    print("="*60)
    print(method_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 2. 情境脆弱度排名 (By Scenario)
    # ---------------------------------------------------------
    scenario_stats = df.groupby("Scenario")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    scenario_stats.columns = ['Scenario', 'Total Attempts', 'Success Count', 'Success Rate']
    scenario_stats['Success Rate'] = (scenario_stats['Success Rate'] * 100).round(2)
    scenario_stats = scenario_stats.sort_values(by="Success Rate", ascending=False)

    print("\n" + "="*60)
    print("🛡️ 各情境 (Scenario) 越獄成功率排名 (越低越安全)")
    print("="*60)
    pd.set_option('display.max_colwidth', 60)
    print(scenario_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 3. 模型安全性排名 (By Model) - 新增功能
    # ---------------------------------------------------------
    model_stats = df.groupby("Model")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    model_stats.columns = ['Target Model', 'Total Attempts', 'Success Count', 'Success Rate']
    model_stats['Success Rate'] = (model_stats['Success Rate'] * 100).round(2)
    model_stats = model_stats.sort_values(by="Success Rate", ascending=True) # 成功率越低越好

    print("\n" + "="*60)
    print("🤖 各模型 (Model) 安全性排名 (Success Rate 越低越好)")
    print("="*60)
    print(model_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 4. SHAP 分群分析 (By Cluster) - 新增功能
    # ---------------------------------------------------------
    # 確保 Cluster 是數值或字串統一，方便排序
    df['Cluster'] = df['Cluster'].astype(str)
    cluster_stats = df.groupby("Cluster")["Success"].agg(['count', 'sum', 'mean']).reset_index()
    cluster_stats.columns = ['SHAP Cluster', 'Total Attempts', 'Success Count', 'Success Rate']
    cluster_stats['Success Rate'] = (cluster_stats['Success Rate'] * 100).round(2)
    cluster_stats = cluster_stats.sort_values(by="Success Rate", ascending=False)

    print("\n" + "="*60)
    print("🧩 SHAP 分群 (Cluster) 越獄模式分析")
    print("各語意分群的攻擊成功率，可對照 shap_analysis_seed1.json")
    print("="*60)
    print(cluster_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 5. 輸出總結
    # ---------------------------------------------------------
    total_attempts = len(df)
    total_success = df["Success"].sum()
    overall_rate = (total_success / total_attempts) * 100

    print("\n" + "="*60)
    print(f"📈 總體統計")
    print(f"總測試次數: {total_attempts}")
    print(f"總成功次數: {total_success}")
    print(f"總體成功率: {overall_rate:.2f}%")
    print("="*60)

    # 儲存詳細統計至 CSV
    output_csv = os.path.join(workspace_root, "experiment_v2", "data", "summary_analysis_full.csv")
    
    # 我們可以將所有維度的統計合併或是儲存最詳細的 Raw Data
    # 這裡選擇儲存 Raw Data 加上統計標記，方便後續用 Excel樞紐分析
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 詳細原始數據已儲存至: {output_csv}")
    print("您可以直接在 Excel 中使用樞紐分析表 (Pivot Table) 進行交叉分析。")

if __name__ == "__main__":
    analyze_results()