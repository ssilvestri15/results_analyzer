#!/usr/bin/env python3
"""
Fix Date Format Script
=====================

Script per identificare e correggere problemi di formato delle date nei file JSON.
Esegui questo script prima di main_workflow.py per risolvere problemi di parsing date.

Usage:
    python fix_date_format.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

def inspect_date_formats(data_dir="data"):
    """
    Ispeziona i formati delle date in tutti i progetti
    """
    print("🔍 INSPECTING DATE FORMATS...")
    print("=" * 50)
    
    data_path = Path(data_dir)
    
    for project_dir in data_path.iterdir():
        if project_dir.is_dir():
            commit_file = project_dir / "commit_metrics.json"
            
            if commit_file.exists():
                print(f"\n📁 Project: {project_dir.name}")
                
                try:
                    with open(commit_file, 'r', encoding='utf-8') as f:
                        commit_data = json.load(f)
                    
                    if commit_data and len(commit_data) > 0:
                        # Esamina le prime 3 date
                        print(f"   Total commits: {len(commit_data)}")
                        
                        for i, commit in enumerate(commit_data[:3]):
                            if 'date' in commit:
                                date_value = commit['date']
                                print(f"   Sample date {i+1}: '{date_value}' (type: {type(date_value).__name__})")
                                
                                # Prova a fare il parsing
                                try:
                                    parsed_date = pd.to_datetime(date_value)
                                    print(f"   ✅ Pandas parsing: SUCCESS -> {parsed_date}")
                                except Exception as e:
                                    print(f"   ❌ Pandas parsing: FAILED -> {str(e)}")
                            else:
                                print(f"   ⚠️  No 'date' field in commit {i+1}")
                    else:
                        print("   ⚠️  Empty commit data")
                        
                except Exception as e:
                    print(f"   ❌ Error reading file: {str(e)}")
            else:
                print(f"   ⚠️  No commit_metrics.json found")

def fix_date_formats(data_dir="data", backup=True):
    """
    Corregge automaticamente i formati delle date
    """
    print("\n🔧 FIXING DATE FORMATS...")
    print("=" * 50)
    
    data_path = Path(data_dir)
    fixed_projects = []
    failed_projects = []
    
    for project_dir in data_path.iterdir():
        if project_dir.is_dir():
            commit_file = project_dir / "commit_metrics.json"
            
            if commit_file.exists():
                print(f"\n📁 Processing: {project_dir.name}")
                
                try:
                    # Leggi i dati originali
                    with open(commit_file, 'r', encoding='utf-8') as f:
                        commit_data = json.load(f)
                    
                    # Backup se richiesto
                    if backup:
                        backup_file = project_dir / "commit_metrics_backup.json"
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            json.dump(commit_data, f, indent=2)
                        print(f"   💾 Backup created: {backup_file.name}")
                    
                    # Correggi le date
                    fixed_count = 0
                    for commit in commit_data:
                        if 'date' in commit:
                            original_date = commit['date']
                            
                            try:
                                # Prova diversi formati
                                if isinstance(original_date, str):
                                    # Formato ISO standard
                                    if 'T' in original_date:
                                        parsed_date = pd.to_datetime(original_date, utc=True)
                                    else:
                                        # Altri formati comuni
                                        parsed_date = pd.to_datetime(original_date)
                                    
                                    # Converti in formato ISO standard
                                    commit['date'] = parsed_date.strftime('%Y-%m-%dT%H:%M:%S%z')
                                    if not commit['date'].endswith('+00:00') and not commit['date'].endswith('Z'):
                                        commit['date'] = parsed_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
                                    
                                    fixed_count += 1
                                    
                            except Exception as e:
                                print(f"   ⚠️  Failed to fix date '{original_date}': {str(e)}")
                    
                    # Salva i dati corretti
                    with open(commit_file, 'w', encoding='utf-8') as f:
                        json.dump(commit_data, f, indent=2)
                    
                    print(f"   ✅ Fixed {fixed_count} dates")
                    fixed_projects.append(project_dir.name)
                    
                except Exception as e:
                    print(f"   ❌ Failed to process: {str(e)}")
                    failed_projects.append(project_dir.name)
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Fixed projects: {len(fixed_projects)}")
    print(f"   ❌ Failed projects: {len(failed_projects)}")
    
    if fixed_projects:
        print(f"   Fixed: {', '.join(fixed_projects)}")
    if failed_projects:
        print(f"   Failed: {', '.join(failed_projects)}")

def test_pandas_parsing(data_dir="data"):
    """
    Testa se le date ora vengono parsate correttamente da pandas
    """
    print("\n🧪 TESTING PANDAS PARSING...")
    print("=" * 50)
    
    data_path = Path(data_dir)
    
    for project_dir in data_path.iterdir():
        if project_dir.is_dir():
            commit_file = project_dir / "commit_metrics.json"
            
            if commit_file.exists():
                print(f"\n📁 Testing: {project_dir.name}")
                
                try:
                    with open(commit_file, 'r', encoding='utf-8') as f:
                        commit_data = json.load(f)
                    
                    # Crea un DataFrame di test
                    df = pd.DataFrame(commit_data[:5])  # Solo prime 5 righe per test
                    
                    # Testa la conversione datetime
                    df['date'] = pd.to_datetime(df['date'])
                    df['period'] = df['date'].dt.to_period('M')
                    
                    print(f"   ✅ SUCCESS - Pandas parsing works!")
                    print(f"   Sample parsed dates:")
                    for i, row in df.iterrows():
                        print(f"      {row['date']} -> Period: {row['period']}")
                        if i >= 2:  # Solo prime 3
                            break
                    
                except Exception as e:
                    print(f"   ❌ FAILED - {str(e)}")

def main():
    """
    Esegue la procedura completa di fix delle date
    """
    print("🛠️  ML Code Smells - Date Format Fixer")
    print("=" * 60)
    
    # Step 1: Inspect current formats
    inspect_date_formats()
    
    # Step 2: Ask user confirmation
    print("\n" + "=" * 60)
    response = input("🤔 Do you want to fix the date formats? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', 'si', 's']:
        # Step 3: Fix the formats
        fix_date_formats(backup=True)
        
        # Step 4: Test the fixes
        test_pandas_parsing()
        
        print("\n🎉 DATE FORMAT FIXING COMPLETED!")
        print("You can now run: python main_workflow.py")
        
    else:
        print("❌ Operation cancelled. No changes made.")

if __name__ == "__main__":
    main()