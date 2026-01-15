import json
import glob
import os
import pandas as pd
import re
import time
import hashlib
from collections import Counter, defaultdict

# --- CONFIGURATION ---
INPUT_DIR = "experiment-analysis/sessions/"
OUTPUT_DIR = "experiment-analysis/output/"

def ensure_directories():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created input directory: {INPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# --- UTILS ---
def get_prompt_hash(system_prompt):
    """Generates a short hash to identify unique system prompts."""
    if not system_prompt: return "No-Prompt"
    return hashlib.md5(system_prompt.encode('utf-8')).hexdigest()[:8]

def extract_roc_auc(tool_output):
    """Parses ROC AUC from the training tool output string."""
    if not isinstance(tool_output, str): return None
    match = re.search(r"ROC AUC = ([\d\.]+)", tool_output)
    return float(match.group(1)) if match else None

def determine_run_status(auc):
    if auc is None: return "❌ Error"
    if auc < 0.4: return "⚠️ Anti-Learning" # Signal in sideband
    if auc < 0.6: return "📉 Random"        # Classifier failed
    if auc > 0.8: return "✅ SUCCESS"       # Anomaly found
    return "❓ Weak"

def parse_tool_calls(messages):
    """
    Robustly scans a message list for tool calls and their outputs.
    Handles both 'id' based matching (OpenAI) and 'name' based matching (Local/Ollama).
    """
    tool_stats = Counter()
    tool_details = []
    
    for i, msg in enumerate(messages):
        # 1. Capture Tool Call
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                t_name = call["function"]["name"]
                t_args = call["function"]["arguments"]
                t_id = call.get("id") # Safe .get() to avoid KeyError
                
                if isinstance(t_args, str):
                    try: t_args = json.loads(t_args)
                    except: t_args = {}
                
                tool_stats[t_name] += 1
                
                # 2. Look ahead for output
                output = "No output found"
                # Scan next few messages for the tool response
                for j in range(i + 1, min(i + 5, len(messages))):
                    next_msg = messages[j]
                    if next_msg.get("role") == "tool":
                        # Match Strategy A: ID Match (Strongest)
                        if t_id and next_msg.get("tool_call_id") == t_id:
                            output = next_msg.get("content", "")
                            break
                        
                        # Match Strategy B: Name Match (Your Framework)
                        # Your logs often use 'tool_name' in the response message
                        if next_msg.get("tool_name") == t_name or next_msg.get("name") == t_name:
                            output = next_msg.get("content", "")
                            break
                        
                        # Match Strategy C: Proximity (Fallback)
                        # If it's the very next message and is a tool, assume it's the response
                        if j == i + 1:
                            output = next_msg.get("content", "")
                            break

                tool_details.append({
                    "name": t_name,
                    "args": t_args,
                    "output": output
                })
    return tool_stats, tool_details

def analyze_session(filepath):
    """Deep analysis of a single session file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        return None

    session_id = os.path.basename(filepath).replace("session_history_", "").replace(".json", "")
    
    # -- 1. Session Metadata --
    model_name = data.get("agent_config", {}).get("base_model", "Unknown Model")
    
    # -- 2. Agent Analysis --
    agents_data = {}
    
    for agent_name, messages in data.get("agents", {}).items():
        # Get System Prompt
        sys_prompt = "None"
        if messages and len(messages) > 0 and messages[0].get("role") == "system":
            sys_prompt = messages[0]["content"]
        
        # Get Tool Usage
        stats, details = parse_tool_calls(messages)
        
        agents_data[agent_name] = {
            "system_prompt_hash": get_prompt_hash(sys_prompt),
            "system_prompt_preview": sys_prompt[:100] + "...",
            "tool_counts": stats,
            "tool_details": details
        }

    # -- 3. Physics / LaCATHODE Specifics --
    physics_runs = []
    analytics = agents_data.get("AnalyticsAgent", {})
    
    if analytics:
        for tool in analytics.get("tool_details", []):
            if tool["name"] == "lacathode_training_tool":
                run_id = tool["args"].get("run_id", "unknown")
                auc = extract_roc_auc(tool["output"])
                status = determine_run_status(auc)
                
                # Try to find associated prep parameters by looking back at previous tool calls
                # (This is an approximation)
                
                physics_runs.append({
                    "Session": session_id,
                    "Run ID": run_id,
                    "Model": model_name,
                    "Prompt Hash": analytics["system_prompt_hash"],
                    "Epochs": f"{tool['args'].get('epochs_flow')}/{tool['args'].get('epochs_clf')}",
                    "ROC AUC": auc,
                    "Status": status
                })

    return {
        "id": session_id,
        "model": model_name,
        "orchestrator": agents_data.get("OrchestratorAgent", {}),
        "analyst": agents_data.get("AnalyticsAgent", {}),
        "physics_runs": physics_runs
    }

def generate_report(sessions, output_dir):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = time.strftime("%Y%m%d_%H%M%S")
    
    # -- AGGREGATE DATA --
    all_physics = []
    tool_matrix = []
    prompt_performance = defaultdict(list)
    model_performance = defaultdict(list)
    failure_modes = Counter()

    for s in sessions:
        all_physics.extend(s["physics_runs"])
        
        # Tool Usage Matrix Row
        row = {"Session": s["id"], "Model": s["model"]}
        total_counts = Counter()
        if s["orchestrator"]: total_counts += s["orchestrator"].get("tool_counts", Counter())
        if s["analyst"]: total_counts += s["analyst"].get("tool_counts", Counter())
        
        row.update(total_counts)
        tool_matrix.append(row)

        # Performance Aggregation
        # We only count sessions that actually RAN a physics training
        if s["physics_runs"]:
            max_auc = max([r["ROC AUC"] for r in s["physics_runs"] if r["ROC AUC"] is not None], default=0)
            
            p_hash = s["analyst"].get("system_prompt_hash", "None")
            prompt_performance[p_hash].append(max_auc)
            model_performance[s["model"]].append(max_auc)

            # Failure Analysis
            for r in s["physics_runs"]:
                failure_modes[r["Status"]] += 1

    # -- WRITE MARKDOWN --
    report_file = os.path.join(output_dir, f"comprehensive_report_{file_ts}.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 📊 LaCATHODE Framework Analytics Report\n")
        f.write(f"**Generated:** {timestamp} | **Sessions Analyzed:** {len(sessions)}\n\n")

        # --- SECTION 1: LEADERBOARDS ---
        f.write("## 🏆 1. Performance Leaderboards\n")
        
        # Model Table
        f.write("### A. Model Efficiency (Based on Training AUC)\n")
        model_data = []
        for m, aucs in model_performance.items():
            if not aucs: continue
            avg_auc = sum(aucs)/len(aucs)
            best_auc = max(aucs)
            success_count = sum(1 for a in aucs if a > 0.8)
            model_data.append({"Model": m, "Sessions w/ Train": len(aucs), "Avg AUC": avg_auc, "Best AUC": best_auc, "Successes": success_count})
        
        if model_data:
            f.write(pd.DataFrame(model_data).to_markdown(index=False, floatfmt=".4f"))
        else:
            f.write("_No models have successfully run training yet._")
        f.write("\n\n")

        # Prompt Table
        f.write("### B. Analyst Prompt Strategy\n")
        prompt_data = []
        for p, aucs in prompt_performance.items():
            if not aucs: continue
            avg_auc = sum(aucs)/len(aucs)
            prompt_data.append({"Prompt Hash": p, "Sessions": len(aucs), "Avg AUC": avg_auc, "Best AUC": max(aucs)})
        
        if prompt_data:
            f.write(pd.DataFrame(prompt_data).to_markdown(index=False, floatfmt=".4f"))
        else:
            f.write("_No prompt data available._")
        f.write("\n\n")

        # --- SECTION 2: FAILURE ANALYSIS ---
        f.write("## 📉 2. Failure Mode Analysis\n")
        if failure_modes:
            df_fail = pd.DataFrame(list(failure_modes.items()), columns=["Status/Diagnosis", "Count"])
            f.write(df_fail.to_markdown(index=False))
        else:
            f.write("No failure modes recorded.")
        f.write("\n\n")

        # --- SECTION 3: PHYSICS RESULTS ---
        f.write("## ⚛️ 3. Physics & Training Log\n")
        if all_physics:
            df_phy = pd.DataFrame(all_physics)
            cols = ["Session", "Run ID", "Status", "ROC AUC", "Epochs", "Model", "Prompt Hash"]
            f.write(df_phy[cols].to_markdown(index=False, floatfmt=".4f"))
        else:
            f.write("❌ No training runs found in these sessions.")
        f.write("\n\n")

        # --- SECTION 4: TOOL USAGE HEATMAP ---
        f.write("## 🛠️ 4. Tool Usage Statistics\n")
        if tool_matrix:
            df_tools = pd.DataFrame(tool_matrix).fillna(0)
            for c in df_tools.columns:
                if c not in ["Session", "Model"]:
                    df_tools[c] = df_tools[c].astype(int)
            f.write(df_tools.to_markdown(index=False))
        f.write("\n\n")

        # --- SECTION 5: SESSION DETAILS ---
        f.write("## 📝 5. Session Drill-Down\n")
        for s in sessions:
            f.write(f"### Session: `{s['id']}`\n")
            f.write(f"- **Model:** {s['model']}\n")
            
            # Orchestrator Stats
            orch = s['orchestrator']
            if orch:
                f.write(f"- **Orchestrator:** Prompt `{orch.get('system_prompt_hash')}` | Tools Used: {dict(orch.get('tool_counts', {}))}\n")
            
            # Analyst Stats
            ana = s['analyst']
            if ana:
                f.write(f"- **Analyst:** Prompt `{ana.get('system_prompt_hash')}` | Tools Used: {dict(ana.get('tool_counts', {}))}\n")
            
            # Physics Outcome Summary
            runs = [r for r in s['physics_runs']]
            if runs:
                best = max([r['ROC AUC'] or 0 for r in runs])
                f.write(f"- **Physics Outcome:** {len(runs)} training runs. Best AUC: **{best:.4f}**\n")
            else:
                f.write(f"- **Physics Outcome:** No training performed.\n")
            
            f.write("---\n")

    return report_file

def main():
    ensure_directories()
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    files.sort(key=os.path.getmtime)
    
    print(f"🔍 Analyzing {len(files)} sessions in {INPUT_DIR}...")
    
    analyzed_sessions = []
    for f in files:
        data = analyze_session(f)
        if data:
            analyzed_sessions.append(data)
            print(f"  -> Parsed {os.path.basename(f)}")
    
    if analyzed_sessions:
        report = generate_report(analyzed_sessions, OUTPUT_DIR)
        print(f"\n✅ FULL REPORT GENERATED: {report}")
        print("Open the Markdown file to view the tables.")
    else:
        print("No valid sessions found.")

if __name__ == "__main__":
    main()